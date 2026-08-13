#include "task_sequence_optimizer_native.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>


namespace {

constexpr int kCategoryCount = TASK_SEQUENCE_CATEGORY_COUNT;
constexpr int kMaxSourcesPerCategory = TASK_SEQUENCE_MAX_SOURCES_PER_CATEGORY;
constexpr int kMaxDpStateCount = 1 << kMaxSourcesPerCategory;
constexpr int kMaxTargetCount = TASK_SEQUENCE_MAX_TARGET_COUNT;
constexpr int kMaxBeamWidth = TASK_SEQUENCE_MAX_BEAM_WIDTH;
constexpr double kInfinity = std::numeric_limits<double>::infinity();


struct SearchInput {
    int target_count;
    int source_count;
    int beam_width;
    uint64_t first_layer_mask;
    const uint64_t* unlock_masks;
    const int32_t* target_categories;
    const int32_t* category_source_counts;
    const int32_t* category_sources;
    const int32_t* source_ids;
    const double* edge_cost_seconds;

    int predecessor_index(int previous_target) const {
        return previous_target < 0 ? target_count : previous_target;
    }

    double edge_cost(int predecessor, int source, int target) const {
        const size_t offset = (
            (static_cast<size_t>(predecessor) * static_cast<size_t>(source_count)
             + static_cast<size_t>(source))
            * static_cast<size_t>(target_count)
            + static_cast<size_t>(target)
        );
        return edge_cost_seconds[offset];
    }

    int category_source(int category, int local_source) const {
        return category_sources[
            category * kMaxSourcesPerCategory + local_source
        ];
    }
};


struct BeamNode {
    uint64_t placed_mask = 0;
    uint64_t available_mask = 0;
    int8_t last_target = -1;
    uint8_t depth = 0;
    std::array<uint8_t, kMaxTargetCount> target_sequence{};
    std::array<double, kCategoryCount * kMaxDpStateCount> category_dp{};
    std::array<double, kCategoryCount> category_best{};
    double prefix_score = 0.0;

    double* category_dp_data(int category) {
        return category_dp.data() + category * kMaxDpStateCount;
    }

    const double* category_dp_data(int category) const {
        return category_dp.data() + category * kMaxDpStateCount;
    }
};


bool lexicographically_smaller(
    const std::array<uint8_t, kMaxTargetCount>& left,
    const std::array<uint8_t, kMaxTargetCount>& right,
    int length
) {
    for (int index = 0; index < length; ++index) {
        if (left[index] != right[index]) {
            return left[index] < right[index];
        }
    }
    return false;
}


bool node_better(const BeamNode& left, const BeamNode& right) {
    if (left.prefix_score != right.prefix_score) {
        return left.prefix_score < right.prefix_score;
    }
    return lexicographically_smaller(
        left.target_sequence,
        right.target_sequence,
        left.depth
    );
}


struct BetterNodeComparator {
    bool operator()(const BeamNode& left, const BeamNode& right) const {
        // std::heap 在该比较器下把最差节点放在堆顶。
        return node_better(left, right);
    }
};


bool candidate_better_than_node(
    double candidate_score,
    const BeamNode& parent,
    int appended_target,
    const BeamNode& other
) {
    if (candidate_score != other.prefix_score) {
        return candidate_score < other.prefix_score;
    }
    for (int index = 0; index < parent.depth; ++index) {
        if (parent.target_sequence[index] != other.target_sequence[index]) {
            return parent.target_sequence[index] < other.target_sequence[index];
        }
    }
    return appended_target < other.target_sequence[parent.depth];
}


struct CategoryDpUpdate {
    std::array<double, kMaxDpStateCount> values{};
    double best_cost = kInfinity;
};


CategoryDpUpdate extend_category_dp(
    const SearchInput& input,
    const BeamNode& parent,
    int category,
    int predecessor,
    int target
) {
    CategoryDpUpdate update;
    update.values.fill(kInfinity);
    const int source_count = input.category_source_counts[category];
    const int state_count = 1 << source_count;
    const int full_source_mask = state_count - 1;
    const double* old_dp = parent.category_dp_data(category);
    for (int used_mask = 0; used_mask < state_count; ++used_mask) {
        const double old_cost = old_dp[used_mask];
        if (!std::isfinite(old_cost)) {
            continue;
        }
        int unused_mask = full_source_mask ^ used_mask;
        while (unused_mask != 0) {
            const int source_bit = unused_mask & -unused_mask;
            int local_source = 0;
            for (int shifted = source_bit; shifted > 1; shifted >>= 1) {
                ++local_source;
            }
            const int source = input.category_source(category, local_source);
            const int new_mask = used_mask | source_bit;
            const double candidate_cost = (
                old_cost + input.edge_cost(predecessor, source, target)
            );
            if (candidate_cost < update.values[new_mask]) {
                update.values[new_mask] = candidate_cost;
            }
            unused_mask ^= source_bit;
        }
    }
    update.best_cost = *std::min_element(
        update.values.begin(),
        update.values.begin() + state_count
    );
    return update;
}


void retain_candidate(
    std::vector<BeamNode>* heap,
    int capacity,
    const BeamNode& parent,
    int target,
    int category,
    const CategoryDpUpdate& dp_update,
    double score,
    uint64_t new_placed,
    uint64_t new_available
) {
    const bool has_capacity = static_cast<int>(heap->size()) < capacity;
    if (!has_capacity && !candidate_better_than_node(
        score,
        parent,
        target,
        heap->front()
    )) {
        return;
    }

    BeamNode child = parent;
    child.placed_mask = new_placed;
    child.available_mask = new_available;
    child.last_target = static_cast<int8_t>(target);
    child.target_sequence[parent.depth] = static_cast<uint8_t>(target);
    child.depth = static_cast<uint8_t>(parent.depth + 1);
    std::copy(
        dp_update.values.begin(),
        dp_update.values.end(),
        child.category_dp_data(category)
    );
    child.category_best[category] = dp_update.best_cost;
    child.prefix_score = score;

    BetterNodeComparator comparator;
    if (has_capacity) {
        heap->push_back(std::move(child));
        std::push_heap(heap->begin(), heap->end(), comparator);
        return;
    }
    std::pop_heap(heap->begin(), heap->end(), comparator);
    heap->back() = std::move(child);
    std::push_heap(heap->begin(), heap->end(), comparator);
}


std::vector<BeamNode> run_beam_search(
    const SearchInput& input,
    TaskSequenceNativeStatisticsV2* statistics
) {
    BeamNode root;
    root.available_mask = input.first_layer_mask;
    root.category_dp.fill(kInfinity);
    root.category_best.fill(0.0);
    for (int category = 0; category < kCategoryCount; ++category) {
        root.category_dp_data(category)[0] = 0.0;
    }

    std::vector<BeamNode> current_layer;
    current_layer.reserve(input.beam_width);
    current_layer.push_back(root);
    statistics->peak_retained_node_count = 1;

    for (int depth = 0; depth < input.target_count; ++depth) {
        std::vector<BeamNode> next_layer;
        next_layer.reserve(input.beam_width);
        for (const BeamNode& parent : current_layer) {
            ++statistics->expanded_parent_count;
            uint64_t available = parent.available_mask & ~parent.placed_mask;
            const int predecessor = input.predecessor_index(parent.last_target);
            while (available != 0) {
                const uint64_t target_bit = available & (~available + 1ULL);
                int target = 0;
                for (uint64_t shifted = target_bit; shifted > 1; shifted >>= 1) {
                    ++target;
                }
                const int category = input.target_categories[target];
                const CategoryDpUpdate dp_update = extend_category_dp(
                    input,
                    parent,
                    category,
                    predecessor,
                    target
                );
                if (std::isfinite(dp_update.best_cost)) {
                    const double score = (
                        parent.prefix_score
                        - parent.category_best[category]
                        + dp_update.best_cost
                    );
                    const uint64_t new_placed = parent.placed_mask | target_bit;
                    const uint64_t new_available = (
                        parent.available_mask
                        | input.unlock_masks[target]
                    ) & ~new_placed;
                    retain_candidate(
                        &next_layer,
                        input.beam_width,
                        parent,
                        target,
                        category,
                        dp_update,
                        score,
                        new_placed,
                        new_available
                    );
                    ++statistics->generated_child_count;
                }
                available ^= target_bit;
            }
        }
        if (next_layer.empty()) {
            throw std::runtime_error(
                "搜索中途没有合法目标或可用实体，已终止原生规划"
            );
        }
        std::sort(next_layer.begin(), next_layer.end(), node_better);
        current_layer = std::move(next_layer);
        statistics->peak_retained_node_count = std::max<uint64_t>(
            statistics->peak_retained_node_count,
            current_layer.size()
        );
    }
    statistics->final_candidate_count = current_layer.size();
    return current_layer;
}


struct AssignmentState {
    double cost = kInfinity;
    uint8_t length = 0;
    std::array<int32_t, kMaxSourcesPerCategory> source_ids{};
    std::array<int16_t, kMaxSourcesPerCategory> source_indices{};
};


bool source_ids_lexicographically_smaller(
    const AssignmentState& left,
    const AssignmentState& right
) {
    for (int index = 0; index < left.length; ++index) {
        if (left.source_ids[index] != right.source_ids[index]) {
            return left.source_ids[index] < right.source_ids[index];
        }
    }
    return false;
}


bool assignment_state_better(
    const AssignmentState& left,
    const AssignmentState& right
) {
    if (left.cost != right.cost) {
        return left.cost < right.cost;
    }
    return source_ids_lexicographically_smaller(left, right);
}


struct FinalCandidate {
    std::array<uint8_t, kMaxTargetCount> target_sequence{};
    std::array<int16_t, kMaxTargetCount> source_sequence{};
    double prefix_score = 0.0;
    double assignment_score = 0.0;
};


FinalCandidate reconstruct_assignment(
    const SearchInput& input,
    const BeamNode& node
) {
    FinalCandidate result;
    result.target_sequence = node.target_sequence;
    result.source_sequence.fill(-1);
    result.prefix_score = node.prefix_score;

    for (int category = 0; category < kCategoryCount; ++category) {
        std::array<int, kMaxSourcesPerCategory> slot_steps{};
        std::array<int, kMaxSourcesPerCategory> slot_predecessors{};
        std::array<int, kMaxSourcesPerCategory> slot_targets{};
        int slot_count = 0;
        int previous_target = -1;
        for (int step = 0; step < input.target_count; ++step) {
            const int target = node.target_sequence[step];
            if (input.target_categories[target] == category) {
                slot_steps[slot_count] = step;
                slot_predecessors[slot_count] = input.predecessor_index(previous_target);
                slot_targets[slot_count] = target;
                ++slot_count;
            }
            previous_target = target;
        }
        if (slot_count == 0) {
            continue;
        }

        const int local_source_count = input.category_source_counts[category];
        const int state_count = 1 << local_source_count;
        const int full_mask = state_count - 1;
        std::array<AssignmentState, kMaxDpStateCount> states{};
        states[0].cost = 0.0;
        for (int slot = 0; slot < slot_count; ++slot) {
            std::array<AssignmentState, kMaxDpStateCount> new_states{};
            for (int used_mask = 0; used_mask < state_count; ++used_mask) {
                if (!std::isfinite(states[used_mask].cost)) {
                    continue;
                }
                int unused_mask = full_mask ^ used_mask;
                while (unused_mask != 0) {
                    const int source_bit = unused_mask & -unused_mask;
                    int local_source = 0;
                    for (int shifted = source_bit; shifted > 1; shifted >>= 1) {
                        ++local_source;
                    }
                    const int source = input.category_source(category, local_source);
                    const int new_mask = used_mask | source_bit;
                    AssignmentState candidate = states[used_mask];
                    candidate.cost += input.edge_cost(
                        slot_predecessors[slot],
                        source,
                        slot_targets[slot]
                    );
                    candidate.source_ids[candidate.length] = input.source_ids[source];
                    candidate.source_indices[candidate.length] = (
                        static_cast<int16_t>(source)
                    );
                    ++candidate.length;
                    if (!std::isfinite(new_states[new_mask].cost)
                        || assignment_state_better(candidate, new_states[new_mask])) {
                        new_states[new_mask] = candidate;
                    }
                    unused_mask ^= source_bit;
                }
            }
            states = new_states;
        }

        const AssignmentState* selected = nullptr;
        for (int mask = 0; mask < state_count; ++mask) {
            if (!std::isfinite(states[mask].cost)) {
                continue;
            }
            if (selected == nullptr || assignment_state_better(states[mask], *selected)) {
                selected = &states[mask];
            }
        }
        if (selected == nullptr || selected->length != slot_count) {
            throw std::runtime_error("原生 source assignment 回溯结果不完整");
        }
        for (int slot = 0; slot < slot_count; ++slot) {
            result.source_sequence[slot_steps[slot]] = selected->source_indices[slot];
        }
    }

    result.assignment_score = 0.0;
    int previous_target = -1;
    for (int step = 0; step < input.target_count; ++step) {
        const int target = result.target_sequence[step];
        const int source = result.source_sequence[step];
        if (source < 0) {
            throw std::runtime_error("原生完整候选缺少 source 对应");
        }
        result.assignment_score += input.edge_cost(
            input.predecessor_index(previous_target),
            source,
            target
        );
        previous_target = target;
    }
    return result;
}


bool final_candidate_better(
    const FinalCandidate& left,
    const FinalCandidate& right,
    const SearchInput& input
) {
    if (left.assignment_score != right.assignment_score) {
        return left.assignment_score < right.assignment_score;
    }
    if (lexicographically_smaller(
        left.target_sequence,
        right.target_sequence,
        input.target_count
    )) {
        return true;
    }
    if (lexicographically_smaller(
        right.target_sequence,
        left.target_sequence,
        input.target_count
    )) {
        return false;
    }
    for (int step = 0; step < input.target_count; ++step) {
        const int32_t left_id = input.source_ids[left.source_sequence[step]];
        const int32_t right_id = input.source_ids[right.source_sequence[step]];
        if (left_id != right_id) {
            return left_id < right_id;
        }
    }
    return false;
}


void write_error(char* buffer, size_t buffer_size, const std::string& message) {
    if (buffer == nullptr || buffer_size == 0) {
        return;
    }
    std::snprintf(buffer, buffer_size, "%s", message.c_str());
}


void validate_input(
    const SearchInput& input,
    int output_candidate_capacity,
    const int32_t* output_candidate_count,
    const int32_t* output_target_sequences,
    const int32_t* output_source_sequences,
    const double* output_prefix_scores,
    const double* output_assignment_scores,
    const TaskSequenceNativeStatisticsV2* output_statistics
) {
    if (input.target_count <= 0 || input.target_count > kMaxTargetCount) {
        throw std::invalid_argument("target_count 必须位于 [1, 63]");
    }
    if (input.source_count <= 0 || input.source_count < input.target_count) {
        throw std::invalid_argument("source_count 必须大于等于 target_count");
    }
    if (input.beam_width <= 0 || input.beam_width > kMaxBeamWidth) {
        throw std::invalid_argument("beam_width 必须位于 [1, 50000]");
    }
    if (output_candidate_capacity <= 0
        || output_candidate_capacity > input.beam_width) {
        throw std::invalid_argument("输出候选容量必须位于 [1, beam_width]");
    }
    if (input.unlock_masks == nullptr
        || input.target_categories == nullptr
        || input.category_source_counts == nullptr
        || input.category_sources == nullptr
        || input.source_ids == nullptr
        || input.edge_cost_seconds == nullptr
        || output_candidate_count == nullptr
        || output_target_sequences == nullptr
        || output_source_sequences == nullptr
        || output_prefix_scores == nullptr
        || output_assignment_scores == nullptr
        || output_statistics == nullptr) {
        throw std::invalid_argument("原生搜索收到空指针参数");
    }

    uint64_t valid_target_mask = (
        (1ULL << static_cast<unsigned>(input.target_count)) - 1ULL
    );
    if ((input.first_layer_mask & ~valid_target_mask) != 0) {
        throw std::invalid_argument("first_layer_mask 包含越界 target 位");
    }
    std::vector<bool> source_seen(input.source_count, false);
    int total_category_sources = 0;
    for (int category = 0; category < kCategoryCount; ++category) {
        const int count = input.category_source_counts[category];
        if (count < 0 || count > kMaxSourcesPerCategory) {
            throw std::invalid_argument("每类 source 数必须位于 [0, 5]");
        }
        total_category_sources += count;
        for (int local_source = 0; local_source < count; ++local_source) {
            const int source = input.category_source(category, local_source);
            if (source < 0 || source >= input.source_count || source_seen[source]) {
                throw std::invalid_argument("类别 source 索引越界或重复");
            }
            source_seen[source] = true;
        }
    }
    if (total_category_sources != input.source_count) {
        throw std::invalid_argument("类别 source 索引未完整覆盖所有实体");
    }
    for (int target = 0; target < input.target_count; ++target) {
        const int category = input.target_categories[target];
        if (category < 0 || category >= kCategoryCount) {
            throw std::invalid_argument("target 类别索引越界");
        }
        if (input.category_source_counts[category] == 0) {
            throw std::invalid_argument("target 类别没有可用 source");
        }
        if ((input.unlock_masks[target] & ~valid_target_mask) != 0) {
            throw std::invalid_argument("unlock_mask 包含越界 target 位");
        }
        for (int local_source = 0;
             local_source < input.category_source_counts[category];
             ++local_source) {
            const int source = input.category_source(category, local_source);
            for (int predecessor = 0;
                 predecessor <= input.target_count;
                 ++predecessor) {
                const double cost = input.edge_cost(predecessor, source, target);
                if (!std::isfinite(cost) || cost < 0.0) {
                    throw std::invalid_argument("匹配类别的边成本必须是有限非负数");
                }
            }
        }
    }
}

}  // namespace


extern "C" TASK_SEQUENCE_API uint32_t task_sequence_optimizer_abi_version(void) {
    return TASK_SEQUENCE_OPTIMIZER_ABI_VERSION;
}


extern "C" TASK_SEQUENCE_API int task_sequence_optimizer_search_v2(
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
) {
    try {
        if (abi_version != TASK_SEQUENCE_OPTIMIZER_ABI_VERSION) {
            throw std::invalid_argument("C++ 任务顺序优化器 ABI 版本不一致");
        }
        SearchInput input{
            target_count,
            source_count,
            beam_width,
            first_layer_mask,
            unlock_masks,
            target_categories,
            category_source_counts,
            category_sources,
            source_ids,
            edge_cost_seconds,
        };
        validate_input(
            input,
            output_candidate_capacity,
            output_candidate_count,
            output_target_sequences,
            output_source_sequences,
            output_prefix_scores,
            output_assignment_scores,
            output_statistics
        );
        *output_candidate_count = 0;
        *output_statistics = TaskSequenceNativeStatisticsV2{};
        if (error_buffer != nullptr && error_buffer_size > 0) {
            error_buffer[0] = '\0';
        }

        const auto beam_started_at = std::chrono::steady_clock::now();
        std::vector<BeamNode> final_nodes = run_beam_search(
            input,
            output_statistics
        );
        const auto beam_finished_at = std::chrono::steady_clock::now();
        output_statistics->beam_search_seconds = (
            std::chrono::duration<double>(beam_finished_at - beam_started_at).count()
        );

        const auto assignment_started_at = std::chrono::steady_clock::now();
        std::vector<FinalCandidate> final_candidates;
        const int returned_candidate_count = std::min<int>(
            output_candidate_capacity,
            final_nodes.size()
        );
        final_candidates.reserve(returned_candidate_count);
        // final_nodes 已按成本和目标序列稳定排序，只回溯前 N 条。
        for (int index = 0; index < returned_candidate_count; ++index) {
            final_candidates.push_back(reconstruct_assignment(
                input,
                final_nodes[index]
            ));
        }
        std::sort(
            final_candidates.begin(),
            final_candidates.end(),
            [&input](const FinalCandidate& left, const FinalCandidate& right) {
                return final_candidate_better(left, right, input);
            }
        );
        const auto assignment_finished_at = std::chrono::steady_clock::now();
        output_statistics->source_assignment_seconds = (
            std::chrono::duration<double>(
                assignment_finished_at - assignment_started_at
            ).count()
        );
        output_statistics->returned_candidate_count = final_candidates.size();

        const int candidate_count = static_cast<int>(final_candidates.size());
        for (int candidate_index = 0;
             candidate_index < candidate_count;
             ++candidate_index) {
            const FinalCandidate& candidate = final_candidates[candidate_index];
            output_prefix_scores[candidate_index] = candidate.prefix_score;
            output_assignment_scores[candidate_index] = candidate.assignment_score;
            for (int step = 0; step < input.target_count; ++step) {
                const size_t output_offset = (
                    static_cast<size_t>(candidate_index)
                    * static_cast<size_t>(input.target_count)
                    + static_cast<size_t>(step)
                );
                output_target_sequences[output_offset] = (
                    candidate.target_sequence[step]
                );
                output_source_sequences[output_offset] = (
                    candidate.source_sequence[step]
                );
            }
        }
        *output_candidate_count = candidate_count;
        return 0;
    } catch (const std::bad_alloc&) {
        write_error(error_buffer, error_buffer_size, "C++ 任务顺序优化器内存不足");
        return -2;
    } catch (const std::exception& error) {
        write_error(error_buffer, error_buffer_size, error.what());
        return -1;
    } catch (...) {
        write_error(error_buffer, error_buffer_size, "C++ 任务顺序优化器发生未知错误");
        return -3;
    }
}
