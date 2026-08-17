// ============================================================================
//  IDBS_PlanA_OpenMP.cpp —— 方案 A（完整去重键）+ 位棋盘 + OpenMP 多核并行
//
//  在 IDBS_PlanA.cpp 基础上，学习 AdvancedIDBS/advance.cpp 的加速技术：
//
//  1) 位棋盘盘面（int grid[][] → unsigned short rows[]）
//     每行一个 unsigned short，10 个 bit 表示 10 列是否被占。
//     can_place / grid_equal / detect_full_rows / estimate 全部改为位运算，
//     盘面比较从 140 个 int 降为 14 个 unsigned short。
//
//  2) OpenMP 多核并行（编译加 -fopenmp 生效）
//     按"源状态"粒度并行展开（schedule(dynamic,64)），每个线程维护自己的
//     本地候选缓冲（按层），最后按层进 critical 区合并去重 + 保 Top-k。
//     未启用 OpenMP 编译时 pragma 被忽略，代码退化为串行且结果一致。
//
//  额外改进：
//  3) 增量满行数：只统计被本次放置"触碰"的行，而不是全盘 14×10 扫描；
//  4) 束宽替换改为"仅当更优才替换"（真正的 Top-k 束），避免原版无条件
//     替换最差状态导致束内混入更差状态；
//  5) 修复 getDistance 的越界隐患：centerCoord 使坐标可能到达 ROWS/COLS
//     之外（如 14.5），原版直接索引 gridCoord[14] 属未定义行为，现钳制到
//     有效格点（可能使个别边界放置的距离值与旧版略有差异）；
//  6) 去重扫描用轻量键（盘面+最后一步+placedBlocks 行），命中才拷贝完整
//     State（约 3KB），避免每次候选都做全量拷贝。
//
//  保留方案 A 的核心：完整去重键（盘面 + 末端位置 + 当前类型已用实例集合），
//  同键保留最短累计距离是严格最优子结构，而非贪心。
//  VALUE_TIEBREAKER 宏：束宽替换同 value 时是否优先淘汰距离更长的。
// ============================================================================

// 跨平台头文件包含
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>  // Linux 动态加载
#include <unistd.h> // Linux 系统调用
#endif

#define BUILD_TEST   // 定义这个宏表示正在构建库
#include "IDBS_head.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdbool.h>
#include <math.h>

// 删除 Windows 特有的安全函数宏
#undef sprintf_s
#undef strcat_s
#define sprintf_s snprintf
#define strcat_s(dest, size, src) strncat(dest, src, size - strlen(dest) - 1)

#define ROWS 14
#define COLS 10
#define MAX_BRICKS 7
#define MAX_ROTATIONS 4
#define MAX_COUNT 5
#define CELL_SIZE 30
#define WIDTH (COLS * CELL_SIZE)
#define HEIGHT (ROWS * CELL_SIZE)
#define k_max 800
#define FULL_ROW_MASK 0x3FF   // 10 位全 1 = 一行满

#define w1 3   //最高高度权重（扣分）
#define w2 55   //空洞块权重（扣分）
#define w3 6   //凹凸起伏权重(扣分)
#define w4 40  //半满行数权重（加分）

// 束宽替换平局规则：1 = 同 value 时优先淘汰距离更长的（优化距离目标）；
//                   0 = 只按 value 替换
#define VALUE_TIEBREAKER 1

// OpenMP 并行（编译加 -fopenmp 生效；未定义 _OPENMP 时 pragma 被忽略，串行执行）
#ifdef _OPENMP
#include <omp.h>
#define IDBS_THREADS omp_get_max_threads()   // 自适应 CPU 逻辑核数，可改为固定值如 14
#endif

typedef struct {
    int dx;
    int dy;
} Node;

typedef struct {
    int id;
    int order;
    int rotation;
    int x;       // 网格坐标x（行）
    int y;       // 网格坐标y（列）
} Step;

// 位棋盘盘面：每行 10 bit（bit j = 第 j 列被占）
typedef struct{
    unsigned short rows[ROWS];
} GridState;

typedef struct{
    Step cur_step[ROWS*COLS];
    int cur_step_len;
    GridState cur_grid_state;
    int value;
    int full_rows;
    int placedBlocks[7][5];
    double distance;
} State;

char brick_name[7][5] = { "LR","LL","ZL","ZR","O","T","Line" };


// 方块定义 [id][rotation][block]
Node bricks[7][4][4] = {
    { // 方块0 (4种形态)
        {{0,0}, {1,0}, {1,1}, {1,2}},
        {{0,0}, {0,1}, {1,0}, {2,0}},
        {{0,0}, {0,1}, {0,2}, {1,2}},
        {{0,0}, {1,0}, {2,0}, {2,-1}}
    },
    { // 方块1 (4种形态)
        {{0,0}, {1,0}, {1,-1}, {1,-2}},
        {{0,0}, {1,0}, {2,0}, {2,1}},
        {{0,0}, {1,0}, {0,1}, {0,2}},
        {{0,0}, {0,1}, {1,1}, {2,1}}
    },
    { // 方块2 (2种形态)
        {{0,0}, {0,1}, {1,1}, {1,2}},
        {{0,0}, {1,0}, {1,-1}, {2,-1}}
    },
    { // 方块3 (2种形态)
        {{0,0}, {0,1}, {1,0}, {1,-1}},
        {{0,0}, {1,0}, {1,1}, {2,1}}
    },
    { // 方块4 (1种形态)
        {{0,0}, {0,1}, {1,0}, {1,1}}
    },
    { // 方块5 (4种形态)
        {{0,0}, {1,0}, {1,-1}, {1,1}},
        {{0,0}, {1,0}, {1,1}, {2,0}},
        {{0,0}, {0,1}, {0,2}, {1,1}},
        {{0,0}, {1,0}, {2,0}, {1,-1}}
    },
    { // 方块6 (2种形态)
        {{0,0}, {0,1}, {0,2}, {0,3}},
        {{0,0}, {1,0}, {2,0}, {3,0}}
    }
};

double centerCoord[7][4][2] = {
    { // 方块0 (4种形态)
        {1.0, 1.0},
        {1.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0}
    },
    { // 方块1 (4种形态)
        {1.0, -1.0},
        {1.0, 0.0},
        {0.0, 1.0},
        {1.0, 1.0}
    },
    { // 方块2 (2种形态)
        {0.5, 1.0},
        {1.0, -0.5}
    },
    { // 方块3 (2种形态)
        {0.5, 0.0},
        {1.0, 0.5}
    },
    { // 方块4 (1种形态)
        {0.5, 0.5}
    },
    { // 方块5 (4种形态)
        {0.5, 0.0},
        {1.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0}
    },
    { // 方块6 (2种形态)
        {0.0, 1.5},
        {1.5, 0.0}
    }
};

time_t start_time;
int beam_width[ROWS]={200,200,200,400,400,400,600,600,600,800,800,800,800,800};
int cur_k;
int rotations[7] = { 4,4,2,2,1,4,2 }; // 方块旋转最大次数
int best_step_count = 0;              // 改为最优步骤计数
int counts[7];                        // 每种方块的总数量
int placedCounts[7];                  // 已放置的方块数量
bool just_satisfied;                  // 理论满行数是否是恰好满行
int theoretical_max_full_rows;        // 理论最大满行数

State layers[ROWS][k_max];            // 分层状态列表，按满行数分层
int layers_size[ROWS];                // 每层状态列表大小
State new_layers[ROWS][k_max];        // 新状态列表，按满行数分层
int new_layers_size[ROWS];            // 每层新状态列表大小

int place_order[7];                   // 放置顺序
int order_index=0;                    // 当前放置顺序索引
int id;                               // 当前方块ID

double shoooting_pose[3];             // 机械臂初始拍摄姿态
// double blockCoord[7][5][2];           // 方块中心抓取坐标
double blockCoord[7][5][2] = {
    {
        {0.912, 2.082},
        {1.362, 2.318},
        {1.094, 1.926},
        {1.224, 2.468},
    },

    {
        {0.936, 2.276},
        {1.344, 2.108},
        {1.186, 1.918},
        {1.056, 2.486},
        {0.884, 2.352}
    },

    {
        {1.338, 2.264},
        {0.918, 2.164},
        {1.168, 1.932},
        {1.074, 2.476},
    },

    {
        {0.928, 2.342},
        {1.356, 2.196},
        {1.246, 1.938},
        {1.024, 2.458},
    },

    {
        {1.326, 2.074},
        {0.902, 2.226},
        {1.214, 1.922},
        {1.088, 2.492},
        {1.354, 2.382}
    },

    {
        {0.946, 2.044},
        {1.332, 2.328},
        {1.142, 1.914},
        {1.252, 2.462},
    },

    {
        {0.896, 2.138},
        {1.348, 2.176},
        {1.202, 1.934},
        {1.042, 2.448},
    }
};
// double gridCoord[ROWS][COLS][2];      // 托盘格点坐标
double gridCoord[ROWS][COLS][2] = {
    {
        {1.000, 2.000}, {1.030, 2.000}, {1.060, 2.000}, {1.090, 2.000}, {1.120, 2.000},
        {1.150, 2.000}, {1.180, 2.000}, {1.210, 2.000}, {1.240, 2.000}, {1.270, 2.000}
    },
    {
        {1.000, 2.030}, {1.030, 2.030}, {1.060, 2.030}, {1.090, 2.030}, {1.120, 2.030},
        {1.150, 2.030}, {1.180, 2.030}, {1.210, 2.030}, {1.240, 2.030}, {1.270, 2.030}
    },
    {
        {1.000, 2.060}, {1.030, 2.060}, {1.060, 2.060}, {1.090, 2.060}, {1.120, 2.060},
        {1.150, 2.060}, {1.180, 2.060}, {1.210, 2.060}, {1.240, 2.060}, {1.270, 2.060}
    },
    {
        {1.000, 2.090}, {1.030, 2.090}, {1.060, 2.090}, {1.090, 2.090}, {1.120, 2.090},
        {1.150, 2.090}, {1.180, 2.090}, {1.210, 2.090}, {1.240, 2.090}, {1.270, 2.090}
    },
    {
        {1.000, 2.120}, {1.030, 2.120}, {1.060, 2.120}, {1.090, 2.120}, {1.120, 2.120},
        {1.150, 2.120}, {1.180, 2.120}, {1.210, 2.120}, {1.240, 2.120}, {1.270, 2.120}
    },
    {
        {1.000, 2.150}, {1.030, 2.150}, {1.060, 2.150}, {1.090, 2.150}, {1.120, 2.150},
        {1.150, 2.150}, {1.180, 2.150}, {1.210, 2.150}, {1.240, 2.150}, {1.270, 2.150}
    },
    {
        {1.000, 2.180}, {1.030, 2.180}, {1.060, 2.180}, {1.090, 2.180}, {1.120, 2.180},
        {1.150, 2.180}, {1.180, 2.180}, {1.210, 2.180}, {1.240, 2.180}, {1.270, 2.180}
    },
    {
        {1.000, 2.210}, {1.030, 2.210}, {1.060, 2.210}, {1.090, 2.210}, {1.120, 2.210},
        {1.150, 2.210}, {1.180, 2.210}, {1.210, 2.210}, {1.240, 2.210}, {1.270, 2.210}
    },
    {
        {1.000, 2.240}, {1.030, 2.240}, {1.060, 2.240}, {1.090, 2.240}, {1.120, 2.240},
        {1.150, 2.240}, {1.180, 2.240}, {1.210, 2.240}, {1.240, 2.240}, {1.270, 2.240}
    },
    {
        {1.000, 2.270}, {1.030, 2.270}, {1.060, 2.270}, {1.090, 2.270}, {1.120, 2.270},
        {1.150, 2.270}, {1.180, 2.270}, {1.210, 2.270}, {1.240, 2.270}, {1.270, 2.270}
    },
    {
        {1.000, 2.300}, {1.030, 2.300}, {1.060, 2.300}, {1.090, 2.300}, {1.120, 2.300},
        {1.150, 2.300}, {1.180, 2.300}, {1.210, 2.300}, {1.240, 2.300}, {1.270, 2.300}
    },
    {
        {1.000, 2.330}, {1.030, 2.330}, {1.060, 2.330}, {1.090, 2.330}, {1.120, 2.330},
        {1.150, 2.330}, {1.180, 2.330}, {1.210, 2.330}, {1.240, 2.330}, {1.270, 2.330}
    },
    {
        {1.000, 2.360}, {1.030, 2.360}, {1.060, 2.360}, {1.090, 2.360}, {1.120, 2.360},
        {1.150, 2.360}, {1.180, 2.360}, {1.210, 2.360}, {1.240, 2.360}, {1.270, 2.360}
    },
    {
        {1.000, 2.390}, {1.030, 2.390}, {1.060, 2.390}, {1.090, 2.390}, {1.120, 2.390},
        {1.150, 2.390}, {1.180, 2.390}, {1.210, 2.390}, {1.240, 2.390}, {1.270, 2.390}
    }
};


// 检查方块是否可以放置（位棋盘版）
bool can_place(int id, int rot, int x, int y, GridState grid_state) {
    for (int i = 0; i < 4; i++) {
        int nx = x + bricks[id][rot][i].dx;
        int ny = y + bricks[id][rot][i].dy;
        if (nx < 0 || nx >= ROWS || ny < 0 || ny >= COLS)
            return false;
        if (grid_state.rows[nx] & (1 << ny))
            return false;
    }
    return true;
}

//计算方块中心坐标
void calculate_center(int id, int rot, int grid_x, int grid_y, float* center_x, float* center_y) {
    int min_dx = 0, max_dx = 0;
    int min_dy = 0, max_dy = 0;

    // 查找最小/最大偏移量
    for (int i = 0; i < 4; i++) {
        int dx = bricks[id][rot][i].dx;
        int dy = bricks[id][rot][i].dy;
        if (dx < min_dx) min_dx = dx;
        if (dx > max_dx) max_dx = dx;
        if (dy < min_dy) min_dy = dy;
        if (dy > max_dy) max_dy = dy;
    }

    // 转换为左下角坐标系
    float base_x = grid_y + 0.5f;           // 列坐标转x轴
    float base_y = ROWS - 1 - grid_x + 0.5f;// 行坐标转y轴（原点在左下）

    // 计算包围盒中心
    *center_x = base_x + (min_dy + max_dy) / 2.0f;
    *center_y = base_y - (min_dx + max_dx) / 2.0f;
}

//评估当前盘面value，用于后续剪枝（位棋盘版，结果与 int 版完全一致）
int estimate(GridState new_state){
    int value=0;
    int grid_top[COLS] = {0};
    int max_top=0;
    int row_node_nums[ROWS] = {0};
    for (int i=0; i<ROWS-1; i++){
        unsigned short r = new_state.rows[i];
        unsigned short rn = new_state.rows[i+1];
        int cnt = 0;
        for (int j=0; j<COLS; j++){
            int mask = 1 << j;
            if (!(r & mask) && (rn & mask)){
                value -= w2; // 空洞块扣分
            }else{
                cnt++;
            }
            if (r & mask){
                grid_top[j] = i+1;
            }
        }
        row_node_nums[i] = cnt;
    }
    for (int j=0; j<COLS; j++){
        if (grid_top[j] > max_top) max_top = grid_top[j];
    }

    //处理最高高度扣分项
    value -= w1*max_top;
    //处理凹凸起伏扣分项
    for (int i=1; i<COLS; i++){
        value -= w3*abs(grid_top[i]-grid_top[i-1]);
    }
    for (int i=0; i<ROWS; i++){
        if (row_node_nums[i] > COLS/2){
            value += w4; // 半满行加分
        }
    }


    return value;
}

//判断盘面是否相同（位棋盘版：14 个 unsigned short 比较）
bool grid_equal(GridState a, GridState b){
    for (int i=0; i<ROWS; i++){
        if (a.rows[i] != b.rows[i]) return false;
    }
    return true;
}

//检测满行数（位棋盘版：一行 10 bit 全 1 即满行）
int detect_full_rows(GridState grid_state){
    int full_rows=0;
    for (int i=0; i<ROWS; i++){
        if (grid_state.rows[i] == FULL_ROW_MASK)
            full_rows++;
    }
    return full_rows;
}

// 完整状态键（方案 A 核心）：盘面 + 最后一步(决定机械臂末端位置)
//                            + 当前类型已用实例集合 placedBlocks[cur_id]
// 未来路径代价只由 (盘面, 末端位置, 剩余实例集合) 决定，
// 同键保留最短累计距离是严格的最优子结构，而不是贪心。
bool state_key_equal(State a, State b, int cur_id){
    if (a.cur_step_len != b.cur_step_len) return false;
    if (a.cur_step_len > 0){
        Step *sa = &a.cur_step[a.cur_step_len - 1];
        Step *sb = &b.cur_step[b.cur_step_len - 1];
        if (sa->id != sb->id || sa->rotation != sb->rotation || sa->x != sb->x || sa->y != sb->y) return false;
    }
    for (int k = 0; k < MAX_COUNT; k++){
        if (a.placedBlocks[cur_id][k] != b.placedBlocks[cur_id][k]) return false;
    }
    return grid_equal(a.cur_grid_state, b.cur_grid_state);
}

// 轻量键比较：候选（盘面 ng + 即将追加的最后一步 + placedBlocks[id] 行）
// vs 已存状态。用于并行版去重扫描，避免为每个候选拷贝完整 State。
bool cand_key_equal(GridState ng, int step_id, int step_rot, int step_x, int step_y,
                    const int placed_row[MAX_COUNT], const State* s, int cur_id){
    if (s->cur_step_len <= 0) return false;
    const Step* last = &s->cur_step[s->cur_step_len - 1];
    if (last->id != step_id || last->rotation != step_rot || last->x != step_x || last->y != step_y) return false;
    for (int k = 0; k < MAX_COUNT; k++){
        if (placed_row[k] != s->placedBlocks[cur_id][k]) return false;
    }
    return grid_equal(ng, s->cur_grid_state);
}

// 索引钳制（修复原版 getDistance 越界隐患）
static inline int clamp_idx(int v, int max_idx){
    return v < 0 ? 0 : (v > max_idx ? max_idx : v);
}

// 网格坐标(可能带 .5 分数) → 物理坐标（四邻域/两点/单点插值），索引越界时钳制
static inline void grid_coord_to_xy(double gx, double gy, double* px, double* py){
    int ix = (int)gx, iy = (int)gy;
    bool fx = (ix < gx), fy = (iy < gy);
    if (fx && fy){
        ix = clamp_idx(ix, ROWS-2); iy = clamp_idx(iy, COLS-2);
        *px = (gridCoord[ix][iy][0] + gridCoord[ix+1][iy][0] + gridCoord[ix][iy+1][0] + gridCoord[ix+1][iy+1][0]) / 4;
        *py = (gridCoord[ix][iy][1] + gridCoord[ix+1][iy][1] + gridCoord[ix][iy+1][1] + gridCoord[ix+1][iy+1][1]) / 4;
    } else if (fx){
        ix = clamp_idx(ix, ROWS-2); iy = clamp_idx(iy, COLS-1);
        *px = (gridCoord[ix][iy][0] + gridCoord[ix+1][iy][0]) / 2;
        *py = (gridCoord[ix][iy][1] + gridCoord[ix+1][iy][1]) / 2;
    } else if (fy){
        ix = clamp_idx(ix, ROWS-1); iy = clamp_idx(iy, COLS-2);
        *px = (gridCoord[ix][iy][0] + gridCoord[ix][iy+1][0]) / 2;
        *py = (gridCoord[ix][iy][1] + gridCoord[ix][iy+1][1]) / 2;
    } else {
        ix = clamp_idx(ix, ROWS-1); iy = clamp_idx(iy, COLS-1);
        *px = gridCoord[ix][iy][0];
        *py = gridCoord[ix][iy][1];
    }
}

// 机械臂单步移动距离 = 末端→抓取点 + 抓取点→放置点
double getDistance(State new_state, int id, int blockIndex, int rot, int row, int col)
{
    double dist=0;
    double preXCoord, preYCoord;
    double targetBlockXCoord = blockCoord[id][blockIndex][0];
    double targetBlockYCoord = blockCoord[id][blockIndex][1];
    double curXCoord, curYCoord;

    if (new_state.cur_step_len == 0){
        preXCoord = shoooting_pose[0];
        preYCoord = shoooting_pose[1];
    }else{
        double preGridXCoord = new_state.cur_step[new_state.cur_step_len-1].x + centerCoord[new_state.cur_step[new_state.cur_step_len-1].id][new_state.cur_step[new_state.cur_step_len-1].rotation][0];
        double preGridYCoord = new_state.cur_step[new_state.cur_step_len-1].y + centerCoord[new_state.cur_step[new_state.cur_step_len-1].id][new_state.cur_step[new_state.cur_step_len-1].rotation][1];
        grid_coord_to_xy(preGridXCoord, preGridYCoord, &preXCoord, &preYCoord);
    }

    double curGridXCoord = row + centerCoord[id][rot][0];
    double curGridYCoord = col + centerCoord[id][rot][1];
    grid_coord_to_xy(curGridXCoord, curGridYCoord, &curXCoord, &curYCoord);

    dist = sqrt((targetBlockXCoord-preXCoord)*(targetBlockXCoord-preXCoord) + (targetBlockYCoord-preYCoord)*(targetBlockYCoord-preYCoord)) + 
           sqrt((curXCoord-targetBlockXCoord)*(curXCoord-targetBlockXCoord) + (curYCoord-targetBlockYCoord)*(curYCoord-targetBlockYCoord));

    return dist;
}


/*
分层迭代束搜索 IDBS（Iterative Depth Beam Search）主函数（OpenMP 并行版）
*/
bool idbs(){
    //查找当前要求顺序下下一个需摆放的方块id
    while (order_index<7 && placedCounts[place_order[order_index]] >= counts[place_order[order_index]]) {
        order_index++;
    }
    if (order_index >= 7) {
        return false; // 没有更多的方块可放
    }
    id = place_order[order_index];

    //初始化新分层状态队列，用于在最后更新旧分层状态队列
    memset(new_layers, 0, sizeof(new_layers));
    memset(new_layers_size, 0, sizeof(new_layers_size));

    // 先统计总状态数，用于动态调度
    int total_states = 0;
    for (int layer=0; layer<ROWS; layer++) total_states += layers_size[layer];
    if (total_states == 0) return false;

    // 并行化：按"源状态"粒度并行（无 OpenMP 编译时 pragma 被忽略，串行执行）
    #pragma omp parallel num_threads(IDBS_THREADS)
    {
        // 线程本地缓冲（按层），避免锁竞争
        State* local_buffer[ROWS];
        int local_size[ROWS] = {0};
        int local_capacity[ROWS] = {0};
        for (int l=0; l<ROWS; l++) {
            local_capacity[l] = beam_width[l] > 100 ? 100 : beam_width[l];
            local_buffer[l] = (State*)malloc(local_capacity[l] * sizeof(State));
        }

        #pragma omp for schedule(dynamic, 64)
        for (int idx=0; idx<total_states; idx++) {
            // 反推 layer 和 i
            int layer = 0;
            int i = idx;
            while (layer < ROWS-1 && i >= layers_size[layer]) {
                i -= layers_size[layer++];
            }
            if (layer >= ROWS || i >= layers_size[layer]) continue;

            // 源状态每输入状态冷拷贝一次（必需），之后所有拷贝都在栈上热拷贝
            State cur_state = layers[layer][i];
            for (int col=0; col<COLS; col++){
                for (int row=ROWS-1; row>-1; row--){
                    if (cur_state.cur_grid_state.rows[row] & (1 << col)) continue;
                    if (row != 0 && !(cur_state.cur_grid_state.rows[row-1] & (1 << col))) continue;
                    for (int rot=0; rot < rotations[id]; rot++){
                        if (!can_place(id, rot, row, col, cur_state.cur_grid_state)) continue;

                        // 轻量构建新盘面（28字节拷贝），记录触碰行
                        GridState ng = cur_state.cur_grid_state;
                        int touched[4];
                        int touched_cnt = 0;
                        for (int k=0; k<4; k++){
                            int nx = row + bricks[id][rot][k].dx;
                            int ny = col + bricks[id][rot][k].dy;
                            ng.rows[nx] |= (1 << ny);
                            bool seen = false;
                            for (int t=0; t<touched_cnt; t++){
                                if (touched[t] == nx){ seen = true; break; }
                            }
                            if (!seen) touched[touched_cnt++] = nx;
                        }

                        // just_satisfied 列高剪枝
                        if (just_satisfied){
                            bool too_high = false;
                            for (int j=0; j<COLS; j++){
                                for (int ii=ROWS-1; ii>=0; ii--){
                                    if (ng.rows[ii] & (1 << j)){
                                        if (ii+1 > theoretical_max_full_rows){ too_high = true; break; }
                                        break;
                                    }
                                }
                                if (too_high) break;
                            }
                            if (too_high) continue;
                        }

                        int value = estimate(ng);

                        // 增量满行数：只统计被触碰行中新增的满行
                        int fr = cur_state.full_rows;
                        for (int t=0; t<touched_cnt; t++){
                            int r = touched[t];
                            if (ng.rows[r] == FULL_ROW_MASK && cur_state.cur_grid_state.rows[r] != FULL_ROW_MASK) fr++;
                        }

                        // 对每个未放置的同类型实例计算距离并插入（完整键去重）
                        for (int blockIndex=0; blockIndex<counts[id]; blockIndex++){
                            if (cur_state.placedBlocks[id][blockIndex] == 1) continue;

                            int placed_row[MAX_COUNT];
                            for (int k=0; k<MAX_COUNT; k++) placed_row[k] = cur_state.placedBlocks[id][k];
                            placed_row[blockIndex] = 1;

                            double dist = cur_state.distance + getDistance(cur_state, id, blockIndex, rot, row, col);

                            // 本地去重（只和本线程同层缓冲比较，用轻量键，命中才拷贝完整 State）
                            int dup_idx = -1;
                            for (int k=0; k<local_size[fr]; k++){
                                if (cand_key_equal(ng, id, rot, row, col, placed_row, &local_buffer[fr][k], id)){
                                    dup_idx = k; break;
                                }
                            }
                            if (dup_idx >= 0){
                                if (dist < local_buffer[fr][dup_idx].distance){
                                    State cand = cur_state;
                                    cand.cur_grid_state = ng; cand.value = value; cand.full_rows = fr; cand.distance = dist;
                                    cand.placedBlocks[id][blockIndex] = 1;
                                    Step step = {id, blockIndex, rot, row, col};
                                    cand.cur_step[cand.cur_step_len++] = step;
                                    local_buffer[fr][dup_idx] = cand;
                                }
                                continue;
                            }

                            // 本地保 Top-k（确认需要保留时才拷贝完整 State）
                            int bk = beam_width[fr];
                            if (local_size[fr] < bk){
                                if (local_size[fr] >= local_capacity[fr]) {
                                    local_capacity[fr] *= 2;
                                    if (local_capacity[fr] > bk) local_capacity[fr] = bk;
                                    local_buffer[fr] = (State*)realloc(local_buffer[fr], local_capacity[fr] * sizeof(State));
                                }
                                State* s = &local_buffer[fr][local_size[fr]++];
                                *s = cur_state;
                                s->cur_grid_state = ng; s->value = value; s->full_rows = fr; s->distance = dist;
                                s->placedBlocks[id][blockIndex] = 1;
                                Step step = {id, blockIndex, rot, row, col};
                                s->cur_step[s->cur_step_len++] = step;
                            } else {
                                // 找最差（value 为主；VALUE_TIEBREAKER=1 时同值平局看距离）
                                int min_i = 0;
                                for (int k=1; k<local_size[fr]; k++){
#if VALUE_TIEBREAKER
                                    if (local_buffer[fr][k].value < local_buffer[fr][min_i].value ||
                                       (local_buffer[fr][k].value == local_buffer[fr][min_i].value &&
                                        local_buffer[fr][k].distance > local_buffer[fr][min_i].distance)) min_i = k;
#else
                                    if (local_buffer[fr][k].value < local_buffer[fr][min_i].value) min_i = k;
#endif
                                }
                                // 仅当新状态严格更优才替换（真正的 Top-k 束）
#if VALUE_TIEBREAKER
                                if (value > local_buffer[fr][min_i].value ||
                                   (value == local_buffer[fr][min_i].value && dist < local_buffer[fr][min_i].distance)){
#else
                                if (value > local_buffer[fr][min_i].value){
#endif
                                    State* s = &local_buffer[fr][min_i];
                                    *s = cur_state;
                                    s->cur_grid_state = ng; s->value = value; s->full_rows = fr; s->distance = dist;
                                    s->placedBlocks[id][blockIndex] = 1;
                                    Step step = {id, blockIndex, rot, row, col};
                                    s->cur_step[s->cur_step_len++] = step;
                                }
                            }
                        }
                    }
                }
            }
        }

        // 按层进 critical 合并，减少锁竞争
        for (int l=0; l<ROWS; l++){
            if (local_size[l] == 0) continue;
            #pragma omp critical (merge_layer)
            {
                for (int j=0; j<local_size[l]; j++){
                    State* s = &local_buffer[l][j];
                    // 全局去重（完整键），命中时保留距离更短的
                    bool dup = false;
                    int dup_idx = -1;
                    for (int k=0; k<new_layers_size[l]; k++){
                        if (state_key_equal(*s, new_layers[l][k], id)){
                            dup = true; dup_idx = k; break;
                        }
                    }
                    if (dup){
                        if (s->distance < new_layers[l][dup_idx].distance) new_layers[l][dup_idx] = *s;
                        continue;
                    }

                    int bk = beam_width[l];
                    if (new_layers_size[l] < bk){
                        new_layers[l][new_layers_size[l]++] = *s;
                    } else {
                        int min_i = 0;
                        for (int k=1; k<new_layers_size[l]; k++){
#if VALUE_TIEBREAKER
                            if (new_layers[l][k].value < new_layers[l][min_i].value ||
                               (new_layers[l][k].value == new_layers[l][min_i].value &&
                                new_layers[l][k].distance > new_layers[l][min_i].distance)) min_i = k;
#else
                            if (new_layers[l][k].value < new_layers[l][min_i].value) min_i = k;
#endif
                        }
#if VALUE_TIEBREAKER
                        if (s->value > new_layers[l][min_i].value ||
                           (s->value == new_layers[l][min_i].value && s->distance < new_layers[l][min_i].distance)){
#else
                        if (s->value > new_layers[l][min_i].value){
#endif
                            new_layers[l][min_i] = *s;
                        }
                    }
                }
            }
        }

        // 释放线程本地缓冲
        for (int l=0; l<ROWS; l++) {
            free(local_buffer[l]);
        }
    }

    // 没有新的状态可扩展，直接返回false
    bool has_new_state = false;
    for (int layer=0; layer<ROWS; layer++){
        if (new_layers_size[layer] > 0) {
            has_new_state = true;
            break;
        }
    }
    if (!has_new_state) {
        return false;
    }

    //更新分层状态队列
    for (int layer=0; layer<ROWS; layer++){
        for (int i=0; i<new_layers_size[layer]; i++){
            layers[layer][i] = new_layers[layer][i];
        }
        layers_size[layer] = new_layers_size[layer];
    }

    //已放置方块数量增加
    placedCounts[place_order[order_index]]++;

    return true;
}


/*
以下为被导出为so文件后被调用的接口
*/
// 公共求解主体：依赖已设置的 counts / place_order / blockCoord / gridCoord / shoooting_pose
static char* run_solve(){
    best_step_count = 0;
    memset(placedCounts, 0, sizeof(placedCounts));
    memset(layers, 0, sizeof(layers));
    memset(layers_size, 0, sizeof(layers_size));
    order_index=0;
    id=-1;

    just_satisfied = false;
    int sum=0;
    for (int i=0; i<7; i++){
        sum+= counts[i];
    }
    theoretical_max_full_rows = sum*4/10;

    if (theoretical_max_full_rows*10 == sum*4) {
        just_satisfied = true;
    }

    //初始化初始状态
    State start_state;
    start_state.cur_step_len = 0;
    start_state.value = 0;
    start_state.full_rows = 0;
    start_state.distance = 0;
    memset(start_state.placedBlocks, 0, sizeof(start_state.placedBlocks));
    GridState start_grid_state;
    for (int i = 0; i < ROWS; i++) {
        start_grid_state.rows[i] = 0;
    }
    start_state.cur_grid_state = start_grid_state;
    layers[0][layers_size[0]++] = start_state;

    //循环束搜索，直到所有方块放完或无法继续放置
    bool flag=true;
    while (order_index < 7 && flag) {
        while (order_index<7 && placedCounts[place_order[order_index]] >= counts[place_order[order_index]]) {
            order_index++;
        }
        if (order_index >= 7) {
            break; // 没有更多的方块可放
        }
        id = place_order[order_index];
        flag = idbs();
    }

    int max_full_rows = 0;
    for (int layer=ROWS-1; layer>=0; layer--){
        if (layers_size[layer] > 0) {
            max_full_rows = layer;
            break;
        }
    }

    if (just_satisfied && max_full_rows != theoretical_max_full_rows){
        just_satisfied = false;
        flag = true;
        while (order_index < 7 && flag) {
            while (order_index<7 && placedCounts[place_order[order_index]] >= counts[place_order[order_index]]) {
                order_index++;
            }
            if (order_index >= 7) {
                break; // 没有更多的方块可放
            }
            id = place_order[order_index];
            flag = idbs();
        }
    }

    for (int layer=ROWS-1; layer>=0; layer--){
        if (layers_size[layer] > 0) {
            max_full_rows = layer;
            break;
        }
    }

    //束搜索结束后，查找最终队列中满行数最多的状态作为结果输出
    //（距离优先，同距离比 value）
    State best_state=layers[max_full_rows][0];
    for(int i=1; i<layers_size[max_full_rows]; i++){
        if  (layers[max_full_rows][i].distance < best_state.distance || 
            layers[max_full_rows][i].distance == best_state.distance && layers[max_full_rows][i].value > best_state.value){
            best_state = layers[max_full_rows][i];
        }
    }

    char* result = (char*)malloc(5000);
    if (!result) return NULL;
    result[0] = '\0';
    
    best_step_count = best_state.cur_step_len;
    for (int i = 0; i < best_step_count; i++) {
        float cx, cy;
        calculate_center(best_state.cur_step[i].id, best_state.cur_step[i].rotation,
            best_state.cur_step[i].x, best_state.cur_step[i].y, &cx, &cy);
        float x = 10 - cx;
        float y = 14 - cy;
        float angle = 0;

        /* 新版智能角度计算 */
        int raw_angle = best_state.cur_step[i].rotation * 90;  // 原始角度
        angle = raw_angle % 360;                     // 标准化到0-360
        if (angle > 180) angle -= 360;               // 转换为-180~180范围

        /* 特殊类型角度修正 */
        switch (best_state.cur_step[i].id) {
        case 0:  // LR
        case 1:  // LL
            angle = angle + 180;
            if (angle > 180) angle -= 360;
            if (angle <= -180) angle += 360;
            break;
        case 2:  // ZL
        case 3:  // ZR
        case 6:  // Line
            angle = (best_state.cur_step[i].rotation % 2) * 90;
            break;
        case 5:  // T
            angle = angle + 180;
            if (angle > 180) angle -= 360;
            break;
        case 4:  // O
            angle = 0;
            break;
        }

        /* 智能坐标补偿 */
        if (best_state.cur_step[i].id == 0 || best_state.cur_step[i].id == 1) {
            if (angle == 0) y -= 0.5;
            if (angle == 90) x -= 0.5;
            if (angle == 180) y += 0.5;
            if (angle == -90) x += 0.5;
        }

        /* 输出格式：每个方块 "名称,角度,x,y,方块索引,"，
           方块索引为该方块实例在 blockCoord[id] 世界坐标表中的下标 */
        char buffer[100];
        if (i == 0) {
            snprintf(result, 5000, "%s,%.0f,%.1f,%.1f,%d,", 
                     brick_name[best_state.cur_step[i].id], angle, x, y,
                     best_state.cur_step[i].order);
        } else {
            snprintf(buffer, sizeof(buffer), "%s,%.0f,%.1f,%.1f,%d,", 
                     brick_name[best_state.cur_step[i].id], angle, x, y,
                     best_state.cur_step[i].order);
            strncat(result, buffer, 5000 - strlen(result) - 1);
        }
    }
    
    char num_buffer[20];
    snprintf(num_buffer, sizeof(num_buffer), "%d", detect_full_rows(best_state.cur_grid_state));
    strncat(result, num_buffer, 5000 - strlen(result) - 1);
    
    return result;

}





/*
以下为被导出为so/dll后被调用的接口：
  1) IDBS(blocks, orders)
       兼容旧接口，使用代码内置的默认坐标（blockCoord/gridCoord/shooting_pose）；
  2) IDBS_Config(blocks, orders, block_coord, grid_coord, shooting_pose)
       方块数量[7]、放置顺序[7]、抓取坐标[7][5][2]=70、托盘格点坐标[14][10][2]=280、
       初始拍摄姿态[3] 全部由调用方通过参数传入（见 IDBS_Config_test.py）。
*/
extern "C" API_SYMBOL char* IDBS(int* blocks, int* orders){
    memset(counts, 0, sizeof(counts));
    memset(place_order, 0, sizeof(place_order));
    for (int i=0; i<7; i++){
        counts[i] = blocks[i];
        place_order[i] = orders[i];
    }
    return run_solve();
}

extern "C" API_SYMBOL char* IDBS_Config(const int* blocks, const int* orders,
                                        const double* block_coord,    // [7][5][2] 共70个double
                                        const double* grid_coord,     // [14][10][2] 共280个double
                                        const double* shooting_pose){ // [3]
    memcpy(blockCoord, block_coord, sizeof(blockCoord));
    memcpy(gridCoord, grid_coord, sizeof(gridCoord));
    memcpy(shoooting_pose, shooting_pose, sizeof(shoooting_pose));
    memset(counts, 0, sizeof(counts));
    memset(place_order, 0, sizeof(place_order));
    for (int i=0; i<7; i++){
        counts[i] = blocks[i];
        place_order[i] = orders[i];
    }
    return run_solve();
}


/*
以下为在main函数中测试的代码，导出为so文件并别调用时不会影响功能
（测试实例与 IDBSwithDistance.cpp 完全相同）
*/
int main()
{
    start_time = time(NULL);

    best_step_count = 0;
    memset(counts, 0, sizeof(counts));
    memset(placedCounts, 0, sizeof(placedCounts));
    memset(layers, 0, sizeof(layers));
    memset(layers_size, 0, sizeof(layers_size));
    memset(place_order, 0, sizeof(place_order));
    order_index=0;
    id=-1;

    //方块数量与放置顺序
    counts[0] = 4;   // LR
    counts[1] = 5;   // LL
    counts[2] = 4;   // ZL
    counts[3] = 4;   // ZR
    counts[4] = 5;   // O
    counts[5] = 4;   // T
    counts[6] = 4;   // Line

    place_order[0] = 5;
    place_order[1] = 4;
    place_order[2] = 1;
    place_order[3] = 0;
    place_order[4] = 3;
    place_order[5] = 2;
    place_order[6] = 6;

    just_satisfied = false;

     //输出理论最大满行数
    int sum=0;
    for (int i=0; i<7; i++){
        sum+= counts[i];
    }
    theoretical_max_full_rows = sum*4/10;
    printf("-------------------------------------------------------------\n");
    printf("Theoretical max full rows: %d\n", theoretical_max_full_rows);

    if (theoretical_max_full_rows*10 == sum*4) {
        just_satisfied = true;
    }

    //初始化初始状态队列
    State start_state;
    start_state.cur_step_len = 0;
    start_state.value = 0;
    start_state.full_rows = 0;
    start_state.distance = 0;
    memset(start_state.placedBlocks, 0, sizeof(start_state.placedBlocks));
    GridState start_grid_state;
    for (int i = 0; i < ROWS; i++) {
        start_grid_state.rows[i] = 0;
    }
    start_state.cur_grid_state = start_grid_state;
    layers[0][layers_size[0]++] = start_state;

    //循环束搜索，直到所有方块放完或无法继续放置
    bool flag=true;
    while (order_index < 7 && flag) {
        while (order_index<7 && placedCounts[place_order[order_index]] >= counts[place_order[order_index]]) {
            order_index++;
        }
        if (order_index >= 7) {
            break; // 没有更多的方块可放
        }
        id = place_order[order_index];
        flag = idbs();
    }

    int max_full_rows = 0;
    for (int layer=ROWS-1; layer>=0; layer--){
        if (layers_size[layer] > 0) {
            max_full_rows = layer;
            break;
        }
    }

    if (just_satisfied && max_full_rows != theoretical_max_full_rows){
        just_satisfied = false;
        printf("恰好满足条件未达成，取消限制继续搜索...\n");
        flag = true;
        while (order_index < 7 && flag) {
            while (order_index<7 && placedCounts[place_order[order_index]] >= counts[place_order[order_index]]) {
                order_index++;
            }
            if (order_index >= 7) {
                break; // 没有更多的方块可放
            }
            id = place_order[order_index];
            flag = idbs();
        }
    }

    for (int layer=ROWS-1; layer>=0; layer--){
        if (layers_size[layer] > 0) {
            max_full_rows = layer;
            break;
        }
    }    

    //束搜索结束后，查找最终队列中满行数最多的状态作为结果输出
    //（距离优先，同距离比 value）
    State best_state=layers[max_full_rows][0];
    for(int i=1; i<layers_size[max_full_rows]; i++){
        if  (layers[max_full_rows][i].distance < best_state.distance || 
            layers[max_full_rows][i].distance == best_state.distance && layers[max_full_rows][i].value > best_state.value){
            best_state = layers[max_full_rows][i];
        }
    }

    // 位棋盘不保存方块类型，由最佳路径的放置序列重建"方块id盘面"再打印
    int type_grid[ROWS][COLS];
    for (int i=0; i<ROWS; i++){
        for (int j=0; j<COLS; j++){
            type_grid[i][j] = -1;
        }
    }
    for (int s=0; s<best_state.cur_step_len; s++){
        Step st = best_state.cur_step[s];
        for (int k=0; k<4; k++){
            int nx = st.x + bricks[st.id][st.rotation][k].dx;
            int ny = st.y + bricks[st.id][st.rotation][k].dy;
            type_grid[nx][ny] = st.id;
        }
    }
    printf("\nThe grid of the best solution (block id, -1=empty):\n");
    for (int i=0; i<ROWS; i++){
        for (int j=0; j<COLS; j++){
            printf("%d ", type_grid[i][j]);
        }
        printf("\n");
    }
    printf("Max full rows:             %d\n", max_full_rows);
    printf("Total distance:            %.2f\n", best_state.distance);
    printf("Solving time:              %ds\n", (int)(time(NULL)-start_time));
    printf("-------------------------------------------------------------\n");

    return 0;
}
