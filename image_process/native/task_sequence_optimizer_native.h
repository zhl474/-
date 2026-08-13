#pragma once

#include <stddef.h>
#include <stdint.h>


#if defined(_WIN32)
#define TASK_SEQUENCE_API __declspec(dllexport)
#else
#define TASK_SEQUENCE_API __attribute__((visibility("default")))
#endif


#ifdef __cplusplus
extern "C" {
#endif


enum {
    TASK_SEQUENCE_OPTIMIZER_ABI_VERSION = 2,
    TASK_SEQUENCE_CATEGORY_COUNT = 7,
    TASK_SEQUENCE_MAX_SOURCES_PER_CATEGORY = 5,
    TASK_SEQUENCE_MAX_TARGET_COUNT = 63,
    TASK_SEQUENCE_MAX_BEAM_WIDTH = 50000,
};


typedef struct TaskSequenceNativeStatisticsV2 {
    uint64_t expanded_parent_count;
    uint64_t generated_child_count;
    uint64_t peak_retained_node_count;
    uint64_t final_candidate_count;
    uint64_t returned_candidate_count;
    double beam_search_seconds;
    double source_assignment_seconds;
} TaskSequenceNativeStatisticsV2;


TASK_SEQUENCE_API uint32_t task_sequence_optimizer_abi_version(void);


/*
 * 所有数组均由调用方分配，函数不会把需要跨语言释放的内存返回给 Python。
 * edge_cost_seconds 使用 C 连续布局 [target_count + 1][source_count][target_count]。
 * category_sources 使用固定布局 [7][5]，未使用位置填 -1。
 * 输出序列使用 [output_candidate_capacity][target_count]。
 * output_candidate_capacity 可小于 beam_width，此时搜索仍保留完整
 * Beam，仅对排序最前的输出容量条候选做实体回溯。
 */
TASK_SEQUENCE_API int task_sequence_optimizer_search_v2(
    uint32_t abi_version,
    int32_t target_count,
    int32_t source_count,
    int32_t beam_width,
    uint64_t first_layer_mask,
    const uint64_t* unlock_masks,
    const int32_t* target_categories,
    const int32_t* category_source_counts,
    const int32_t* category_sources,
    const int32_t* source_ids,
    const double* edge_cost_seconds,
    int32_t output_candidate_capacity,
    int32_t* output_candidate_count,
    int32_t* output_target_sequences,
    int32_t* output_source_sequences,
    double* output_prefix_scores,
    double* output_assignment_scores,
    TaskSequenceNativeStatisticsV2* output_statistics,
    char* error_buffer,
    size_t error_buffer_size
);


#ifdef __cplusplus
}
#endif
