#!/bin/bash

# ============================================
# GAR 推理：Qwen3-VL（MC / Simple / Detailed），贪心解码
# 数据见 behaviors.py（../DATA/GAR）
# ============================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

TASK="detailed"                    # mc | simple | detailed

# 生成参数（贪心：do_sample=False，temperature=0；top_p 在贪心下不参与采样）
MAX_NEW_TOKENS=1024
TEMPERATURE=0.0
TOP_P=1.0
SEED=42
MAX_NUM=""                   # 留空=全量；设为正整数则只跑前 N 条，例如 MAX_NUM=100
OUT=""                       # 留空=默认 ../RESULT/gar_qwen3vl/8b/results_gar_*.json

# 可选开关（1 启用）
USE_THINK=0                  # GAR-MC 使用 think 模板
USE_FLASH_ATTN=1
USE_VERBOSE=1                # 1：每条打印 [q]（wrapper）与 [a] 完整原始回复，同 prompting_with_steering.py

# HuggingFace 缓存目录
export HF_HOME=/sda/hf_cache
export TRANSFORMERS_CACHE=/sda/hf_cache
export HUGGINGFACE_HUB_CACHE=/sda/hf_cache

# GPU 配置
export CUDA_VISIBLE_DEVICES=1

# 组装可选 CLI 参数
EXTRA_ARGS=()
if [ -n "${MAX_NUM}" ]; then
    EXTRA_ARGS+=(--max_num "${MAX_NUM}")
fi
if [ -n "${OUT}" ]; then
    EXTRA_ARGS+=(--out "${OUT}")
fi
if [ "${USE_THINK}" = "1" ]; then
    EXTRA_ARGS+=(--think)
fi
if [ "${USE_FLASH_ATTN}" = "1" ]; then
    EXTRA_ARGS+=(--use_flash_attn)
fi
if [ "${USE_VERBOSE}" = "1" ]; then
    EXTRA_ARGS+=(--verbose)
fi

echo "========================================"
echo "GAR: Qwen3-VL"
echo "========================================"
echo "Task: ${TASK}"
echo "Model size: 8b"
echo "Max new tokens: ${MAX_NEW_TOKENS}"
echo "Temperature / top_p: ${TEMPERATURE} / ${TOP_P}"
echo "Seed: ${SEED}"
if [ -n "${MAX_NUM}" ]; then
    echo "Max num samples: ${MAX_NUM}"
else
    echo "Max num samples: (full split)"
fi
if [ -n "${OUT}" ]; then
    echo "Output: ${OUT}"
else
    echo "Output: default under ../RESULT/gar_qwen3vl/8b/"
fi
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Verbose: ${USE_VERBOSE} (1=print [q]/[a] per sample)"
echo "========================================"
echo ""

python run_gar_qwen3vl.py \
    --task "${TASK}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}" \
    --seed "${SEED}" \
    "${EXTRA_ARGS[@]}"

echo ""
echo "========================================"
echo "GAR inference finished."
echo "========================================"
