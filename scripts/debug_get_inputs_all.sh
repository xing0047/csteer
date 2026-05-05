#!/usr/bin/env bash
set -u

# Debug all dataset types by saving model.get_inputs outputs.
# - Runs both ref and noref for each type.
# - Prints progress in bash ([i/total], elapsed, success/fail).
# - Continues on failures and summarizes at the end.
#
# Usage:
#   bash scripts/debug_get_inputs_all.sh
#   MODEL_NAME=qwen3vl MODEL_SIZE=8b OUT_DIR=preview bash scripts/debug_get_inputs_all.sh
#   TYPES="gar_image_mc_qa inst_it_image_mc_qa" bash scripts/debug_get_inputs_all.sh
#
# Notes:
# - This script uses prompting_with_steering.py --debug_get_inputs, so it exits
#   right after model.get_inputs for the first sample.
# - BLINK noref currently has no separate source path; this script still runs it
#   so you can verify current behavior consistency.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}" || exit 1

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_NAME="${MODEL_NAME:-qwen3vl}"
MODEL_SIZE="${MODEL_SIZE:-8b}"
OUT_DIR="${OUT_DIR:-preview}"
VIDEO_MAX_FRAMES="${VIDEO_MAX_FRAMES:-64}"

# Default full type list discussed in this task.
DEFAULT_TYPES=(
  "gar_image_mc_qa"
  "gar_image_detail_oe_qa"
  "gar_image_simple_oe_qa"
  "vip_image_oe_qa"
  "blink_image_mc_qa"
  "inst_it_image_mc_qa"
  "inst_it_image_oe_qa"
  "inst_it_video_mc_qa"
  "inst_it_video_oe_qa"
)

if [[ -n "${TYPES:-}" ]]; then
  # shellcheck disable=SC2206
  TYPES_ARR=(${TYPES})
else
  TYPES_ARR=("${DEFAULT_TYPES[@]}")
fi

MODES=("ref" "noref")
TOTAL=$(( ${#TYPES_ARR[@]} * ${#MODES[@]} ))
DONE=0
OK=0
FAIL=0

START_TS="$(date +%s)"

echo "== Debug get_inputs all =="
echo "ROOT_DIR=${ROOT_DIR}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "MODEL=${MODEL_NAME}_${MODEL_SIZE}"
echo "OUT_DIR=${OUT_DIR}"
echo "VIDEO_MAX_FRAMES=${VIDEO_MAX_FRAMES}"
echo "TOTAL_TASKS=${TOTAL}"
echo

for t in "${TYPES_ARR[@]}"; do
  for mode in "${MODES[@]}"; do
    DONE=$((DONE + 1))
    NOW_TS="$(date +%s)"
    ELAPSED=$((NOW_TS - START_TS))

    TAG="[${DONE}/${TOTAL}]"
    echo "${TAG} type=${t} mode=${mode} elapsed=${ELAPSED}s"

    CMD=(
      "${PYTHON_BIN}" "prompting_with_steering.py"
      "--model_name" "${MODEL_NAME}"
      "--model_size" "${MODEL_SIZE}"
      "--type" "${t}"
      "--debug_get_inputs"
      "--debug_out_dir" "${OUT_DIR}"
      "--debug_video_max_frames" "${VIDEO_MAX_FRAMES}"
    )

    if [[ "${mode}" == "noref" ]]; then
      CMD+=("--noref")
    fi

    if "${CMD[@]}"; then
      OK=$((OK + 1))
      echo "  -> OK"
    else
      FAIL=$((FAIL + 1))
      echo "  -> FAIL (continued)"
    fi
    echo
  done
done

END_TS="$(date +%s)"
TOTAL_ELAPSED=$((END_TS - START_TS))

echo "== Summary =="
echo "Total   : ${TOTAL}"
echo "Success : ${OK}"
echo "Fail    : ${FAIL}"
echo "Elapsed : ${TOTAL_ELAPSED}s"

if [[ "${FAIL}" -gt 0 ]]; then
  exit 1
fi

