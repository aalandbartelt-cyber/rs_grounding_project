#!/usr/bin/env bash
set -euo pipefail

CODE_DIR="$(cd -- "$(dirname -- "$0")/code" && pwd)"
RSG_ROOT="${RSG_ROOT:-$HOME/rsg_work/fast}"
RSG_PYTHON="${RSG_PYTHON:-/tmp/rsg-env/bin/python}"
STAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RSG_RUN_ROOT:-$RSG_ROOT/runs/baseline_${STAMP}}"
ARCHIVE="$RSG_ROOT/data/rs_civilian_reviewed.zip"
WEIGHTS="$RSG_ROOT/weights"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/rsg-triton-cache}"
mkdir -p "$RUN_ROOT" "$TRITON_CACHE_DIR"

# 1. 核对民航数据划分、两套权重及分配的 A100。
"$RSG_PYTHON" "$CODE_DIR/prepare_check.py" --archive "$ARCHIVE" \
  --weights-dir "$WEIGHTS" --models all --cuda --reuse-runtime \
  2>&1 | tee "$RUN_ROOT/01_preflight.log"

# 2. 先跑单图，检查离线推理、候选框和 JSON 输出。
"$RSG_PYTHON" "$CODE_DIR/check_civilian_grounding.py" --archive "$ARCHIVE" \
  --weights-dir "$WEIGHTS" --device cuda --limit 1 \
  --output "$RUN_ROOT/experiments/M_dino_one" 2>&1 | tee "$RUN_ROOT/02_smoke.log"

# 3. 按首轮 A100 参数评估全部 47 张留出图的定位结果。
"$RSG_PYTHON" "$CODE_DIR/check_civilian_grounding.py" --archive "$ARCHIVE" \
  --weights-dir "$WEIGHTS" --device cuda --limit 47 --box-threshold 0.25 \
  --text-threshold 0.25 --max-candidates 20 --output "$RUN_ROOT/experiments/M_dino_val" \
  2>&1 | tee "$RUN_ROOT/03_grounding.log"

# 4. 用同一数据划分复训检索适配层，记录零样本与微调指标。
"$RSG_PYTHON" "$CODE_DIR/train_civilian_retrieval.py" --archive "$ARCHIVE" \
  --weights "$WEIGHTS/chinese-clip-vit-base-patch16" --device cuda \
  --rank 16 --epochs 80 --lr 0.002 --drift-weight 0.25 \
  --output "$RUN_ROOT/experiments/S_clip_a100" 2>&1 | tee "$RUN_ROOT/04_retrieval.log"

# 5. 按满分 1 的公式合并四项指标，并将归档写到侧栏的主目录。
"$RSG_PYTHON" "$CODE_DIR/score_rounds.py" --root "$RUN_ROOT" \
  --zip "$HOME/rsg_a100_baseline_${STAMP}.zip" 2>&1 | tee "$RUN_ROOT/05_score.log"
