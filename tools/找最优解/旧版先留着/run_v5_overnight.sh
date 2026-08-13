#!/usr/bin/env bash
# V5.1 overnight pipeline
#
# 用法：
#   1) 先 cd 到包含以下两个文件的目录：
#        tetris_layout_search_v5.py
#        tetris_layout_search_v5_1.py
#   2) 把本脚本也放在该目录，然后：
#        chmod +x run_v5_overnight.sh
#        ./run_v5_overnight.sh
#
# 推荐在 tmux 里运行，SSH/VSCode 断开不会影响计算。
#
# 流水线：
#   Pass 1: top 7000, K=1, 20s, seed=1      —— 快速铺开覆盖
#   Pass 2: top 7000, K=1, 60s, seed=137    —— 新随机种子重试 stubborn signatures
#   Pass 3: top 2549, K=3, 60s, seed=1001   —— 给最高概率约 80% 区域补空间多样性
#   Final : top 7000, K=1, status-only      —— 重新导出完整 7000 范围 library 并打印最终状态
#
# 所有阶段使用同一个 output-dir，并启用 --resume。
# 每个阶段正常退出后会写 marker；脚本被中断后重新运行时会跳过已完成阶段。

set -Eeuo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
PROGRAM="${PROGRAM:-tetris_layout_search_v5_1.py}"
BASE_PROGRAM="${BASE_PROGRAM:-tetris_layout_search_v5.py}"
OUTPUT_DIR="${OUTPUT_DIR:-layouts_260_v5}"
LOG_DIR="${LOG_DIR:-v5_overnight_logs}"
STATE_DIR="${STATE_DIR:-v5_overnight_state}"

# 默认按 Linux 报告的物理核心数设置 jobs；也可启动前覆盖，例如：
#   JOBS=8 ./run_v5_overnight.sh
if command -v lscpu >/dev/null 2>&1; then
    DETECTED_PHYSICAL_CORES="$(lscpu -p=Core,Socket 2>/dev/null | grep -v '^#' | sort -u | wc -l | tr -d ' ')"
else
    DETECTED_PHYSICAL_CORES="$(nproc)"
fi
JOBS="${JOBS:-$DETECTED_PHYSICAL_CORES}"
CP_WORKERS="${CP_WORKERS:-1}"
HEARTBEAT="${HEARTBEAT:-30}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR" "$STATE_DIR"

RUN_ID="$(date '+%Y%m%d_%H%M%S')"
MASTER_LOG="$LOG_DIR/overnight_${RUN_ID}.log"

PASS1_MARKER="$STATE_DIR/pass1_k1_20s.done"
PASS2_MARKER="$STATE_DIR/pass2_k1_60s.done"
PASS3_MARKER="$STATE_DIR/pass3_top2549_k3_60s.done"

log() {
    printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$MASTER_LOG"
}

run_stage() {
    local stage_name="$1"
    local stage_log="$2"
    local marker="$3"
    shift 3

    if [[ -n "$marker" && -f "$marker" ]]; then
        log "SKIP $stage_name：检测到已完成 marker $marker"
        return 0
    fi

    log "============================================================"
    log "START $stage_name"
    log "stage log: $stage_log"
    log "command: $*"

    # pipefail 保证 Python 异常时整个 stage 返回非 0，set -e 会停止后续流水线。
    PYTHONUNBUFFERED=1 "$@" 2>&1 | tee -a "$stage_log" "$MASTER_LOG"

    if [[ -n "$marker" ]]; then
        printf 'completed_at=%s\n' "$(date '+%F %T')" > "$marker"
    fi
    log "END   $stage_name"
}

# ---------- preflight ----------
log "V5.1 overnight pipeline starting"
log "cwd=$(pwd)"
log "output_dir=$OUTPUT_DIR"
log "jobs=$JOBS, cp_workers=$CP_WORKERS, heartbeat=${HEARTBEAT}s"
log "detected_physical_cores=$DETECTED_PHYSICAL_CORES"

if [[ ! -f "$PROGRAM" ]]; then
    log "ERROR: 当前目录找不到 $PROGRAM"
    exit 2
fi
if [[ ! -f "$BASE_PROGRAM" ]]; then
    log "ERROR: 当前目录找不到 $BASE_PROGRAM；V5.1 需要和 V5 放在同一目录"
    exit 2
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    log "ERROR: 找不到 Python：$PYTHON_BIN"
    exit 2
fi
if ! "$PYTHON_BIN" -c 'import ortools' >/dev/null 2>&1; then
    log "ERROR: 当前 Python 环境无法 import ortools"
    exit 2
fi

log "preflight OK"

# ---------- Pass 1 ----------
run_stage \
    "PASS 1 / 快速 K=1 覆盖：top7000, 20s, seed=1" \
    "$LOG_DIR/pass1_k1_20s_${RUN_ID}.log" \
    "$PASS1_MARKER" \
    "$PYTHON_BIN" "$PROGRAM" \
        --output-dir "$OUTPUT_DIR" \
        --top-signatures 7000 \
        --layouts-per-signature 1 \
        --jobs "$JOBS" \
        --cp-workers "$CP_WORKERS" \
        --task-time 20 \
        --seed 1 \
        --heartbeat "$HEARTBEAT" \
        --resume

# ---------- Pass 2 ----------
run_stage \
    "PASS 2 / stubborn K=1 重试：top7000, 60s, seed=137" \
    "$LOG_DIR/pass2_k1_60s_${RUN_ID}.log" \
    "$PASS2_MARKER" \
    "$PYTHON_BIN" "$PROGRAM" \
        --output-dir "$OUTPUT_DIR" \
        --top-signatures 7000 \
        --layouts-per-signature 1 \
        --jobs "$JOBS" \
        --cp-workers "$CP_WORKERS" \
        --task-time 60 \
        --seed 137 \
        --heartbeat "$HEARTBEAT" \
        --resume

# ---------- Pass 3 ----------
run_stage \
    "PASS 3 / 高概率区域补 K=3：top2549, 60s, seed=1001" \
    "$LOG_DIR/pass3_top2549_k3_60s_${RUN_ID}.log" \
    "$PASS3_MARKER" \
    "$PYTHON_BIN" "$PROGRAM" \
        --output-dir "$OUTPUT_DIR" \
        --top-signatures 2549 \
        --layouts-per-signature 3 \
        --jobs "$JOBS" \
        --cp-workers "$CP_WORKERS" \
        --task-time 60 \
        --seed 1001 \
        --heartbeat "$HEARTBEAT" \
        --resume

# ---------- Final status / full library re-export ----------
# Pass 3 会按 top2549 导出 library，因此最后必须用 top7000 再 status-only 一次，
# 把 library_shard00of01.jsonl 恢复为完整 top7000 范围。
log "============================================================"
log "START FINAL STATUS / top7000 K=1 + full library re-export"
PYTHONUNBUFFERED=1 "$PYTHON_BIN" "$PROGRAM" \
    --output-dir "$OUTPUT_DIR" \
    --top-signatures 7000 \
    --layouts-per-signature 1 \
    --jobs "$JOBS" \
    --cp-workers "$CP_WORKERS" \
    --task-time 60 \
    --seed 137 \
    --heartbeat "$HEARTBEAT" \
    --resume \
    --status-only \
    2>&1 | tee -a "$LOG_DIR/final_status_${RUN_ID}.log" "$MASTER_LOG"
log "END FINAL STATUS"

log "============================================================"
log "OVERNIGHT PIPELINE COMPLETE"
log "完整结果目录：$OUTPUT_DIR"
log "主日志：$MASTER_LOG"
log "Pass 3 日志可查看 top2549 的 K=3 完成情况"
log "最终 library：$OUTPUT_DIR/library_shard00of01.jsonl"
