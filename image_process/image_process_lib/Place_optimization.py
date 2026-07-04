import numpy as np
from scipy.optimize import linear_sum_assignment
import time

def _check_finite_points(points, name, type_idx):
    arr = np.array(points, dtype=np.float64)
    if arr.size == 0:
        return
    if not np.all(np.isfinite(arr)):
        invalid_pos = np.argwhere(~np.isfinite(arr))
        raise ValueError(f"{name} 第 {type_idx} 类包含无效数值，位置: {invalid_pos.tolist()}")

def optimize_block_assignment(blocks, targets, list):
    """
    优化7种方块类型的分配问题，精确处理方块数量
    
    输入：
    - blocks: 包含7个子列表的列表，每个子列表包含n_i个方块，每个方块有[x, y, z, theta]
    - targets: 包含7个子列表的列表，每个子列表包含m_i个目标点，每个目标点有[x, y]
    
    输出：
    - block2: 优化后的方块列表（保持原始数量，仅重新排序）
    - assignment_info: 分配信息字典，包含被选中的方块和对应关系
    - distance_info: 距离信息字典，包含详细距离统计
    - time_info: 时间信息字典，包含处理时间
    """
    # 验证输入结构
    assert len(blocks) == 7, "blocks必须包含7种方块类型"
    assert len(targets) == 7, "targets必须包含7种目标点类型"
    
    # 初始化结果
    block2 = [None] * 7
    assignment_info = [{"selected": [], "mapping": {}} for _ in range(7)]
    original_distances = np.zeros(7)
    optimized_distances = np.zeros(7)
    distance_saving = np.zeros(7)
    time_used = np.zeros(7)

    # 记录开始时间
    total_start_time = time.time()
    
    # 处理每种方块类型
    for type_idx in range(7):
        type_start_time = time.time()
        
        # 获取当前类型的方块和目标点
        type_blocks = blocks[type_idx].copy()  # 创建副本避免修改原始数据
        type_targets = targets[type_idx].copy()
        
        n_blocks = len(type_blocks)
        n_targets = len(type_targets)
        n = min(n_blocks, n_targets)  # 实际可分配的数量
        _check_finite_points(type_blocks, "方块坐标", type_idx)
        _check_finite_points(type_targets, "目标坐标", type_idx)
        
        # 1. 如果没有可分配的点，保持原始顺序
        if n == 0:
            block2[type_idx] = type_blocks
            time_used[type_idx] = (time.time() - type_start_time) * 1000
            continue
        
        # 2. 计算原始总距离（按输入顺序对应）
        original_distance_arrive = 0
        original_distance_return = 0
        for i in range(min(n_blocks, n_targets)):
            dx = type_blocks[i][0] - type_targets[i][0]
            dy = type_blocks[i][1] - type_targets[i][1]
            if type_targets[i][2] == 0:
                return_distance = 0
            else:
                dx_return = type_blocks[i][0] - list[type_targets[i][2] - 1][0]
                dy_return = type_blocks[i][1] - list[type_targets[i][2] - 1][1]
                return_distance = np.hypot(dx_return, dy_return)
            
            #print(f"type {type_idx}: block_x={type_blocks[i][0]:.2f}, target_x={type_targets[i][0]:.2f}, dx={dx:.2f}")
            arrive_distance = np.hypot(dx, dy)
            original_distance_arrive += arrive_distance
            original_distance_return += return_distance
        original_distance = original_distance_arrive + original_distance_return

        # 3. 构建成本矩阵（仅考虑可分配部分）
        cost_matrix = np.zeros((n_blocks, n_targets))
        for i in range(n_blocks):
            for j in range(n_targets):
                dx = type_blocks[i][0] - type_targets[j][0]
                dy = type_blocks[i][1] - type_targets[j][1]
                if type_targets[j][2] == 0:
                    return_distance = 0
                else:
                    dx_return = type_blocks[i][0] - list[type_targets[j][2] - 1][0]
                    dy_return = type_blocks[i][1] - list[type_targets[j][2] - 1][1]
                    return_distance = np.hypot(dx_return, dy_return)

                arrive_distance = np.hypot(dx, dy)
                cost_matrix[i, j] = arrive_distance + return_distance

        if not np.all(np.isfinite(cost_matrix)):
            invalid_pos = np.argwhere(~np.isfinite(cost_matrix))
            raise ValueError(f"代价矩阵第 {type_idx} 类包含无效数值，位置: {invalid_pos.tolist()}")

        # 4. 使用匈牙利算法求解最优匹配
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # 5. 创建优化后的方块顺序（保持原始数量）
        # 5.1 初始化优化后的方块列表（保持原始顺序）
        optimized_blocks = type_blocks.copy()
        
        # 5.2 记录哪些方块被选中
        selected_indices = set()
        mapping = {}
        
        # 5.3 应用优化分配
        for k in range(len(row_ind)):
            i = row_ind[k]
            j = col_ind[k]
            
            # 只处理有效的分配（在可分配范围内）
            if j < n:
                # 记录被选中的方块
                selected_indices.add(i)
                
                # 记录映射关系
                mapping[j] = i
                
                # 将方块移动到对应目标点的位置
                optimized_blocks[j] = type_blocks[i]
        
        # 5.4 处理未选中的方块（保持原位置）
        # 找出所有未选中的方块索引
        unselected_indices = [idx for idx in range(n_blocks) if idx not in selected_indices]
        
        # 将未选中的方块按顺序放在已分配方块之后
        for pos, idx in enumerate(unselected_indices):
            target_pos = n + pos
            if target_pos < n_blocks:
                optimized_blocks[target_pos] = type_blocks[idx]
        
        # 6. 保存结果
        block2[type_idx] = optimized_blocks
        assignment_info[type_idx] = {
            "selected": sorted(selected_indices),
            "unselected": unselected_indices,
            "mapping": mapping
        }
        
        # 7. 计算优化后的总距离
        optimized_distance_arrive = 0
        optimized_distance_return = 0
        for j in range(n):
            if j in mapping:
                i = mapping[j]
                dx = type_blocks[i][0] - type_targets[j][0]
                dy = type_blocks[i][1] - type_targets[j][1]
                optimized_distance_arrive += np.hypot(dx, dy)
                
                # 回程：从当前目标点到下一个方块（通过全局上一个目标点）
                if type_targets[j][2] == 0:
                    return_distance = 0
                else:
                    dx_return = type_blocks[i][0] - list[type_targets[j][2] - 1][0]
                    dy_return = type_blocks[i][1] - list[type_targets[j][2] - 1][1]
                    return_distance = np.hypot(dx_return, dy_return)
                optimized_distance_return += return_distance
        optimized_distance = optimized_distance_arrive + optimized_distance_return
        
        # 8. 保存统计信息
        original_distances[type_idx] = original_distance
        optimized_distances[type_idx] = optimized_distance
        distance_saving[type_idx] = original_distance - optimized_distance
        time_used[type_idx] = (time.time() - type_start_time) * 1000
    
    # 9. 计算总时间和总节省
    total_time = (time.time() - total_start_time) * 1000
    total_original_distance = np.sum(original_distances)
    total_optimized_distance = np.sum(optimized_distances)
    total_saving = total_original_distance - total_optimized_distance
    
    # 10. 打包返回结果
    distance_info = {
        "original_per_type": original_distances,
        "optimized_per_type": optimized_distances,
        "saving_per_type": distance_saving,
        "total_original": total_original_distance,
        "total_optimized": total_optimized_distance,
        "total_saving": total_saving
    }
    
    time_info = {
        "per_type": time_used,
        "total": total_time,
        "average": np.mean(time_used),
        "max": np.max(time_used),
        "min": np.min(time_used)
    }
    
    return (
        block2,  # 优化后的方块数据
        [info["selected"] for info in assignment_info],  # 每种类型选中的方块索引
        distance_info["original_per_type"],  # 每种类型的原始距离
        distance_info["optimized_per_type"],  # 每种类型的优化后距离
        distance_info["saving_per_type"],  # 每种类型的节省距离
        time_info["per_type"],  # 每种类型的处理时间
        time_info["total"],  # 总处理时间
        distance_info["total_original"],  # 原始总距离
        distance_info["total_optimized"]  # 优化后总距离
    )

def print_optimization_results(block2, selected, orig_dists, opt_dists, dist_saving, 
                              time_used, total_time, orig_total, opt_total):
    """打印优化结果（符合要求的格式）"""
    print("\n" + "="*50)
    print("方块分配优化结果")
    print("="*50)
    
    # 打印每种类型的结果
    print("\n【每种方块类型的优化效果】")
    print(f"{'类型':<6}{'方块数':<8}{'选中数':<8}{'原始距离':<12}{'优化后距离':<12}{'节省距离':<12}{'节省百分比':<12}{'处理时间(ms)':<12}")
    for i in range(7):
        n_blocks = len(block2[i])
        n_selected = len(selected[i])
        
        orig_dist = orig_dists[i]
        opt_dist = opt_dists[i]
        saving = dist_saving[i]
        
        if orig_dist > 0:
            saving_percent = (saving / orig_dist) * 100
        else:
            saving_percent = 0.0
            
        print(f"{i:<6}{n_blocks:<8}{n_selected:<8}{orig_dist:.4f}{'':<2}{opt_dist:.4f}{'':<2}{saving:.4f}{'':<4}{saving_percent:>6.1f}%{'':<4}{time_used[i]:.2f}")
    
    # 打印总体结果
    print("\n【总体优化效果】")
    print(f"原始总距离: {orig_total:.4f}")
    print(f"优化后总距离: {opt_total:.4f}")
    print(f"总节省距离: {orig_total - opt_total:.4f}")
    
    if orig_total > 0:
        saving_percent = ((orig_total - opt_total) / orig_total) * 100
    else:
        saving_percent = 0.0
    print(f"节省百分比: {saving_percent:.1f}%")
    print(f"总处理时间: {total_time:.2f} ms")
    
    # 打印时间分析
    if np.any(time_used > 0):
        print("\n【处理时间分析】")
        print(f"平均每类型处理时间: {np.mean(time_used):.2f} ms")
        print(f"最长类型处理时间: {np.max(time_used):.2f} ms")
        print(f"最短类型处理时间: {np.min(time_used):.2f} ms")
    
    # 打印被选中的方块信息
    print("\n【被选中的方块】")
    for i in range(7):
        if selected[i]:
            print(f"类型 {i}: 选中方块索引 {selected[i]}")
    
    return orig_total, opt_total

def get_put_pose(x,y,shooting_angle, calibrator):  #眼
    set_angle=list(shooting_angle)
    test_point=(x,y)
    transformed_point = calibrator.transform(test_point)
    set_angle[0]=transformed_point[0]
    set_angle[1]=transformed_point[1]
    set_angle[2]=0
    return set_angle


def put_fenlei(list,shooting_angle,calibrator):
    out_list=[[],[],[],[],[],[],[]]
    for i in list:
        tool_position=get_put_pose(i[0],i[1],shooting_angle,calibrator)
        j=[tool_position[0],tool_position[1],i[4]]
        if(i[3]=='L_blue'):
            out_list[0].append(j)
        elif(i[3]=='L_yellow'):
            out_list[1].append(j)
        elif(i[3]=='z_blue'):
            out_list[2].append(j)
        elif(i[3]=='z_green'):
            out_list[3].append(j)
        elif(i[3]=='square'):
            out_list[4].append(j)
        elif(i[3]=='T'):
            out_list[5].append(j)
        elif(i[3]=='line'):
            out_list[6].append(j)
    return out_list
def cube_pocess(list):
    out_list=[[],[],[],[],[],[],[]]
    cnt=0
    for i in list:
        for j in i:
            out_list[cnt].append([j[0],j[1],j[2],j[3]])
        cnt=cnt+1
    return out_list
def get_all_cube(category,cube):#按 L_blue L_yellow z_blue z_green square T line 顺序统计
    if(category=="L_blue"):
        cube[0]=cube[0]+1
    elif(category=="L_yellow"):
        cube[1]=cube[1]+1
    elif(category=="z_blue"):
        cube[2]=cube[2]+1
    elif(category=="z_green"):
        cube[3]=cube[3]+1
    elif(category=="square"):
        cube[4]=cube[4]+1
    elif(category=="T"):
        cube[5]=cube[5]+1
    elif(category=="line"):
        cube[6]=cube[6]+1
    return cube

def make_list(l,category,cam_3d,t):#将检测到的方块加入到方块表中
    x,y,z=cam_3d
    if(category=="L_blue"):
        l[0].append([x,y,z,t])
    elif(category=="L_yellow"):
        l[1].append([x,y,z,t])
    elif(category=="z_blue"):
        l[2].append([x,y,z,t])
    elif(category=="z_green"):
        l[3].append([x,y,z,t])
    elif(category=="square"):
        l[4].append([x,y,z,t])
    elif(category=="T"):
        l[5].append([x,y,z,t])
    elif(category=="line"):
        l[6].append([x,y,z,t])
    return l



cube_index=[0,0,0,0,0,0,0]

def get_cube_location(cube,block2,index_flag):#取出这个方块的位置、角度信息
    index=0
    if(cube[3]=="L_blue"):
        index=0
    elif(cube[3]=="L_yellow"):
        index=1
    elif(cube[3]=="z_blue"):
        index=2
    elif(cube[3]=="z_green"):
        index=3
    elif(cube[3]=="square"):
        index=4
    elif(cube[3]=="T"):
        index=5
    elif(cube[3]=="line"):
        index=6
    x=block2[index][cube_index[index]][0]
    y=block2[index][cube_index[index]][1]
    z=block2[index][cube_index[index]][2]
    t=block2[index][cube_index[index]][3]
    if(index_flag):
        cube_index[index]=cube_index[index]+1
    return x,y,z,t
