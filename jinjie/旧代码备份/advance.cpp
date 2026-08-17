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

#define w1 3   //最高高度权重（扣分）
#define w2 55   //空洞块权重（扣分）
#define w3 6   //凹凸起伏权重(扣分)
#define w4 40  //半满行数权重（加分）

//OpenMP 并行方案
#ifdef _OPENMP
#include <omp.h>
#endif

typedef struct {
    int dx;
    int dy;
} Node;

typedef struct {
    int id;
    int rotation;
    int x;       // 网格坐标x（行）
    int y;       // 网格坐标y（列）
} Step;

typedef struct{
    unsigned short rows[ROWS];
} GridState;

typedef struct{
    Step cur_step[ROWS*COLS];
    int cur_step_len;
    GridState cur_grid_state;
    int value;
    int full_rows;
} State;

// 移除 Windows 特有的 #pragma pack
typedef struct {
    char signature[2];
    unsigned int file_size;
    unsigned int reserved;
    unsigned int data_offset;
} BmpFileHeader;

char brick_name[7][10] = { "L_blue","L_yellow","z_blue","z_green","square","T","line" };

typedef struct {
    unsigned int header_size;
    int width;
    int height;
    unsigned short planes;
    unsigned short bits_per_pixel;
    unsigned int compression;
    unsigned int image_size;
    int x_pixels_per_meter;
    int y_pixels_per_meter;
    unsigned int colors_used;
    unsigned int important_colors;
} BmpInfoHeader;

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

int beam_width[ROWS]={200,200,200,400,400,400,600,600,600,800,800,800,800,800};
int cur_k;
int rotations[7] = { 4,4,2,2,1,4,2 }; //方块旋转最大次数
int best_step_count = 0;              // 改为最优步骤计数
int counts[7];                        // 每种方块的剩余数量
bool just_satisfied;                  // 理论满行数是否是恰好满行
int theoretical_max_full_rows;        // 理论最大满行数

State layers[ROWS][k_max];                // 分层状态列表，按满行数分层
int layers_size[ROWS];                // 每层状态列表大小
State new_layers[ROWS][k_max];            // 新状态列表，按满行数分层
int new_layers_size[ROWS];            // 每层新状态列表大小

int place_order[7];                   // 放置顺序
int order_index=0;                    // 当前放置顺序索引
int id;                               // 当前方块ID

// 检查方块是否可以放置
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

//评估当前盘面value，用于后续剪枝
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

//判断盘面是否相同，用于去重
bool grid_equal(GridState a, GridState b){
    for (int i=0; i<ROWS; i++){
        if (a.rows[i] != b.rows[i]) return false;
    }
    return true;
}

//检测满行数
int detect_full_rows(GridState grid_state){
    int full_rows=0;
    for (int i=0; i<ROWS; i++){
        if (grid_state.rows[i] == 0x3FF) // 10位全1 = 满行
            full_rows++;
    }
    return full_rows;
}


//添加新状态至状态队列，并保持束宽度不超过k
void append_state(State cur_state, int id, int rot, int row, int col, State new_layers[ROWS][k_max], int new_layers_size[ROWS]){
    
    State new_state = cur_state;
    for (int i=0; i<4; i++){
        int nx = row + bricks[id][rot][i].dx;
        int ny = col + bricks[id][rot][i].dy;
        new_state.cur_grid_state.rows[nx] |= (1 << ny);
    }

    if (just_satisfied){
        for (int j=0; j<COLS; j++){
            for (int i=ROWS-1; i>=0; i--){
                if (new_state.cur_grid_state.rows[i] & (1 << j)){
                    if (i+1 > theoretical_max_full_rows) return;
                    break;
                }
            }
        }
    }

    new_state.value = estimate(new_state.cur_grid_state);
    Step new_step = {id, rot, row, col};
    new_state.cur_step[new_state.cur_step_len++] = new_step;
    new_state.full_rows = detect_full_rows(new_state.cur_grid_state);

    //若已存在相同盘面，则无需添加
    for(int i=0; i<new_layers_size[new_state.full_rows]; i++){
        if (grid_equal(new_state.cur_grid_state, new_layers[new_state.full_rows][i].cur_grid_state)){
            return;
        }
    }

    cur_k = beam_width[new_state.full_rows];

    //若束宽未满，则直接添加新状态
    if (new_layers_size[new_state.full_rows] < cur_k){
        new_layers[new_state.full_rows][new_layers_size[new_state.full_rows]++] = new_state;
    }else{
        //若束宽已满，则替换价值最低的状态
        int min_index=0;
        for (int i=1; i<new_layers_size[new_state.full_rows]; i++){
            if (new_layers[new_state.full_rows][min_index].value > new_layers[new_state.full_rows][i].value){
                min_index = i;
            }
        }
        new_layers[new_state.full_rows][min_index] = new_state;
    }
}



/*
分层迭代束搜索 IDBS（Iterative Depth Beam Search）主函数
*/
bool idbs(){
    while (order_index<7 && counts[place_order[order_index]] == 0) order_index++;
    if (order_index >= 7) return false;
    id = place_order[order_index];

    memset(new_layers, 0, sizeof(new_layers));
    memset(new_layers_size, 0, sizeof(new_layers_size));

    // 先统计总状态数，用于动态调度
    int total_states = 0;
    int state_offsets[ROWS] = {0};
    for (int layer=0; layer<ROWS; layer++) {
        state_offsets[layer] = total_states;
        total_states += layers_size[layer];
    }
    
    if (total_states == 0) return false;

    // 并行化：按状态粒度并行，而不是按layer
    #pragma omp parallel num_threads(14)
    {
        // 线程本地缓冲区，动态分配避免大静态数组
        State* local_buffer[ROWS];
        int local_size[ROWS] = {0};
        int local_capacity[ROWS] = {0};
        
        // 初始化本地缓冲区
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
                    if (!(cur_state.cur_grid_state.rows[row] & (1 << col))){
                        if (row == 0 || (cur_state.cur_grid_state.rows[row-1] & (1 << col))){
                            for (int rot=0; rot < rotations[id]; rot++){
                                if (can_place(id, rot, row, col, cur_state.cur_grid_state)){
                                    // 轻量构建新盘面（28字节拷贝），此时仍不拷贝步骤历史
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

                                    // just_satisfied 剪枝
                                    if (just_satisfied){
                                        bool too_high = false;
                                        for (int j=0; j<COLS; j++){
                                            for (int ii=ROWS-1; ii>=0; ii--){
                                                if (ng.rows[ii] & (1 << j)){
                                                    if (ii+1 > theoretical_max_full_rows){
                                                        too_high = true;
                                                        break;
                                                    }
                                                    break;
                                                }
                                            }
                                            if (too_high) break;
                                        }
                                        if (too_high) continue;
                                    }

                                    int value = estimate(ng);

                                    // 增量计算满行数：只统计被触碰行中新增的满行
                                    int fr = cur_state.full_rows;
                                    for (int t=0; t<touched_cnt; t++){
                                        int r = touched[t];
                                        if (ng.rows[r] == 0x3FF && cur_state.cur_grid_state.rows[r] != 0x3FF) fr++;
                                    }

                                    // 本地去重（只和本地比较）
                                    bool dup = false;
                                    for (int k=0; k<local_size[fr]; k++){
                                        if (grid_equal(ng, local_buffer[fr][k].cur_grid_state)){
                                            dup = true; break;
                                        }
                                    }
                                    if (dup) continue;

                                    // 本地保Top-k（确认需要保留时才拷贝完整State）
                                    int bk = beam_width[fr];
                                    Step step = {id, rot, row, col};
                                    if (local_size[fr] < bk){
                                        if (local_size[fr] >= local_capacity[fr]) {
                                            local_capacity[fr] = local_capacity[fr] * 2;
                                            if (local_capacity[fr] > bk) local_capacity[fr] = bk;
                                            local_buffer[fr] = (State*)realloc(local_buffer[fr], local_capacity[fr] * sizeof(State));
                                        }
                                        State* s = &local_buffer[fr][local_size[fr]++];
                                        *s = cur_state;
                                        s->cur_grid_state = ng;
                                        s->value = value;
                                        s->full_rows = fr;
                                        s->cur_step[s->cur_step_len++] = step;
                                    } else {
                                        int min_i = 0;
                                        for (int k=1; k<local_size[fr]; k++){
                                            if (local_buffer[fr][min_i].value > local_buffer[fr][k].value) min_i = k;
                                        }
                                        if (value > local_buffer[fr][min_i].value){
                                            State* s = &local_buffer[fr][min_i];
                                            *s = cur_state;
                                            s->cur_grid_state = ng;
                                            s->value = value;
                                            s->full_rows = fr;
                                            s->cur_step[s->cur_step_len++] = step;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // 按层分critical合并，减少锁竞争
        for (int l=0; l<ROWS; l++){
            if (local_size[l] == 0) continue;
            
            #pragma omp critical (merge_layer)
            {
                for (int j=0; j<local_size[l]; j++){
                    State* s = &local_buffer[l][j];
                    bool dup = false;
                    for (int k=0; k<new_layers_size[l]; k++){
                        if (grid_equal(s->cur_grid_state, new_layers[l][k].cur_grid_state)){
                            dup = true; break;
                        }
                    }
                    if (dup) continue;

                    int bk = beam_width[l];
                    if (new_layers_size[l] < bk){
                        new_layers[l][new_layers_size[l]++] = *s;
                    } else {
                        int min_i = 0;
                        for (int k=1; k<new_layers_size[l]; k++){
                            if (new_layers[l][min_i].value > new_layers[l][k].value) min_i = k;
                        }
                        if (s->value > new_layers[l][min_i].value) new_layers[l][min_i] = *s;
                    }
                }
            }
        }
        
        // 释放本地缓冲区
        for (int l=0; l<ROWS; l++) {
            free(local_buffer[l]);
        }
    }

    bool has_new = false;
    for (int l=0; l<ROWS; l++) if (new_layers_size[l] > 0) { has_new = true; break; }
    if (!has_new) return false;

    for (int l=0; l<ROWS; l++){
        for (int i=0; i<new_layers_size[l]; i++) layers[l][i] = new_layers[l][i];
        layers_size[l] = new_layers_size[l];
    }

    counts[place_order[order_index]]--;
    return true;
}

/*
以下为被导出为so文件后被调用的接口
*/
static char* run_idbs(bool with_cells,
                      int block1, int block2, int block3, int block4,
                      int block5, int block6, int block7,
                      int order1, int order2, int order3, int order4,
                      int order5, int order6, int order7){
    best_step_count = 0;
    memset(counts, 0, sizeof(counts));
    memset(layers, 0, sizeof(layers));
    memset(layers_size, 0, sizeof(layers_size));
    memset(place_order, 0, sizeof(place_order));
    order_index=0;
    id=-1;

    counts[0] = block1;
    counts[1] = block2;
    counts[2] = block3;
    counts[3] = block4;
    counts[4] = block5;
    counts[5] = block6;
    counts[6] = block7;
    place_order[0] = order1;
    place_order[1] = order2;
    place_order[2] = order3;
    place_order[3] = order4;
    place_order[4] = order5;
    place_order[5] = order6;
    place_order[6] = order7;

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
    GridState start_grid_state;
    for (int i = 0; i < ROWS; i++) {
        start_grid_state.rows[i] = 0;
    }
    start_state.cur_grid_state = start_grid_state;
    layers[0][layers_size[0]++] = start_state;

    //循环束搜索，直到所有方块放完或无法继续放置
    bool flag=true;
    while (order_index < 7 && flag) {
        while (order_index<7 && counts[place_order[order_index]] == 0) {
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
            while (order_index<7 && counts[place_order[order_index]] == 0) {
                order_index++;
            }
            if (order_index >= 7) {
                break; // 没有更多的方块可放
            }
            id = place_order[order_index];
            flag = idbs();
        }
    }

    //束搜索结束后，查找最终队列中满行数最多的状态作为结果输出
    State best_state=layers[max_full_rows][0];
    for(int i=1; i<layers_size[max_full_rows]; i++){
        if  (layers[max_full_rows][i].value > best_state.value){
            best_state = layers[max_full_rows][i];
        }
    }

    const size_t result_size = 10000;
    char* result = (char*)malloc(result_size);
    if (!result) return NULL;
    result[0] = '\0';
    
    best_step_count = best_state.cur_step_len;
    if (with_cells) {
        snprintf(result, result_size, "IDBS_WITH_CELLS_V1,%d,", best_step_count);
    }
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
        case 0:  // L_blue
        case 1:  // L_yellow
            angle = angle + 180;
            if (angle > 180) angle -= 360;
            if (angle <= -180) angle += 360;
            break;
        case 2:  // z_blue
        case 3:  // z_green
        case 6:  // line
            angle = (best_state.cur_step[i].rotation % 2) * 90;
            break;
        case 5:  // T
            angle = angle + 180;
            if (angle > 180) angle -= 360;
            break;
        case 4:  // square
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

        char buffer[256];
        if (with_cells) {
            int runtime_cells[4][2];
            for (int cell_index = 0; cell_index < 4; cell_index++) {
                int internal_row = best_state.cur_step[i].x
                    + bricks[best_state.cur_step[i].id][best_state.cur_step[i].rotation][cell_index].dx;
                int internal_col = best_state.cur_step[i].y
                    + bricks[best_state.cur_step[i].id][best_state.cur_step[i].rotation][cell_index].dy;
                // 运行时托盘使用左侧为第 1 列、底层为第 1 行。
                runtime_cells[cell_index][0] = COLS - internal_col;
                runtime_cells[cell_index][1] = internal_row + 1;
            }
            snprintf(
                buffer,
                sizeof(buffer),
                "%s,%.0f,%.1f,%.1f,%d,%d,%d,%d,%d,%d,%d,%d,",
                brick_name[best_state.cur_step[i].id],
                angle,
                x + 0.5f,
                y + 0.5f,
                runtime_cells[0][0], runtime_cells[0][1],
                runtime_cells[1][0], runtime_cells[1][1],
                runtime_cells[2][0], runtime_cells[2][1],
                runtime_cells[3][0], runtime_cells[3][1]
            );
            strncat(result, buffer, result_size - strlen(result) - 1);
        } else if (i == 0) {
            snprintf(result, result_size, "%s,%.0f,%.1f,%.1f,",
                     brick_name[best_state.cur_step[i].id], angle, x, y);
        } else {
            snprintf(buffer, sizeof(buffer), "%s,%.0f,%.1f,%.1f,",
                     brick_name[best_state.cur_step[i].id], angle, x, y);
            strncat(result, buffer, result_size - strlen(result) - 1);
        }
    }
    
    char num_buffer[20];
    snprintf(num_buffer, sizeof(num_buffer), "%d", detect_full_rows(best_state.cur_grid_state));
    strncat(result, num_buffer, result_size - strlen(result) - 1);
    
    return result;

}


extern "C" API_SYMBOL char* IDBS(
    int block1, int block2, int block3, int block4, int block5, int block6, int block7,
    int order1, int order2, int order3, int order4, int order5, int order6, int order7){
    return run_idbs(
        false,
        block1, block2, block3, block4, block5, block6, block7,
        order1, order2, order3, order4, order5, order6, order7
    );
}


extern "C" API_SYMBOL char* IDBSWithCells(
    int block1, int block2, int block3, int block4, int block5, int block6, int block7,
    int order1, int order2, int order3, int order4, int order5, int order6, int order7){
    return run_idbs(
        true,
        block1, block2, block3, block4, block5, block6, block7,
        order1, order2, order3, order4, order5, order6, order7
    );
}





/*
以下为在main函数中测试的代码，导出为so文件并别调用时不会影响功能
*/
int main()
{
    best_step_count = 0;
    memset(counts, 0, sizeof(counts));
    memset(layers, 0, sizeof(layers));
    memset(layers_size, 0, sizeof(layers_size));
    memset(place_order, 0, sizeof(place_order));
    order_index=0;
    id=-1;

    //方块数量与放置顺序
    counts[0] = 5;   // LR
    counts[1] = 5;   // LL
    counts[2] = 4;   // ZL
    counts[3] = 5;   // ZR
    counts[4] = 5;   // O
    counts[5] = 4;   // T
    counts[6] = 5;   // Line

    place_order[0] = 2;
    place_order[1] = 6;
    place_order[2] = 5;
    place_order[3] = 0;
    place_order[4] = 3;
    place_order[5] = 1;
    place_order[6] = 4;

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
    GridState start_grid_state;
    for (int i = 0; i < ROWS; i++)
        start_grid_state.rows[i] = 0;
    start_state.cur_grid_state = start_grid_state;
    layers[0][layers_size[0]++] = start_state;

    //循环束搜索，直到所有方块放完或无法继续放置
    bool flag=true;
    while (order_index < 7 && flag) {
        while (order_index<7 && counts[place_order[order_index]] == 0) {
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
            while (order_index<7 && counts[place_order[order_index]] == 0) {
                order_index++;
            }
            if (order_index >= 7) {
                break; // 没有更多的方块可放
            }
            id = place_order[order_index];
            flag = idbs();
        }
    }

    //束搜索结束后，查找最终队列中满行数最多的状态作为结果输出
    State best_state=layers[max_full_rows][0];
    for(int i=1; i<layers_size[max_full_rows]; i++){
        if  (layers[max_full_rows][i].value > best_state.value){
            best_state = layers[max_full_rows][i];
        }
    }

    printf("Max full rows:             %d\n", max_full_rows);
    for (int i=0; i<ROWS; i++){
        for (int j=0; j<COLS; j++){
            printf("%d ", (best_state.cur_grid_state.rows[i] & (1 << j)) ? 1 : 0);
        }
        printf("\n");
    }
    printf("-------------------------------------------------------------\n");

    return 0;
}
