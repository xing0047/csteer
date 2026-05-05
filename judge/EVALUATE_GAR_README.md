# GAR Evaluation Tool

Unified evaluation tool for GAR-Bench formats: GAR-VQA (multi-choice), GAR-Simple, and GAR-Detailed.

## Overview

The `evaluate_gar.py` script provides a unified interface for evaluating GAR-Bench predictions across three different formats:

- **GAR-VQA**: Multi-choice question answering with type-based accuracy statistics
- **GAR-Simple**: Simple format evaluation using vLLM
- **GAR-Detailed**: Detailed format evaluation using vLLM

## Setup

Ensure you have the required dependencies installed:

```bash
pip install openai tqdm
```

For GAR-Simple and GAR-Detailed evaluations, you need a vLLM service running. Start it with:

```bash
# Example: Start vLLM service on port 8001
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-72B-Instruct-AWQ \
    --port 8001
```

## Usage

### Basic Command Structure

```bash
python evaluate_gar.py \
    --eval_type <type> \
    --input_dir <input_directory> \
    --log_dir <output_directory> \
    [--pattern <file_pattern>] \
    [--summary_out <summary_csv>] \
    [--base_url <vllm_url>] \
    [--model_name <model_name>] \
    [--num_process <num>] \
    [--layer_range <min> <max>]
```

### Arguments

- `--eval_type` (required): Evaluation type - `vqa`, `simple`, or `detailed`
- `--input_dir` (required): Directory containing prediction JSON files
- `--log_dir` (required): Directory to save per-file evaluation metrics
- `--pattern` (optional): Glob pattern to match files (auto-detected if not provided)
- `--summary_out` (optional): Path to save summary CSV file
- `--base_url` (optional): vLLM OpenAI API endpoint (default: `http://localhost:8000/v1`)
- `--model_name` (optional): Model name for vLLM evaluator (default: `Qwen/Qwen2.5-72B-Instruct-AWQ`)
- `--num_process` (optional): Number of parallel processes (default: 8)
- `--layer_range` (optional): Only evaluate layers in this range (e.g., `0 35`)

## Evaluation Types

### GAR-VQA (Multi-Choice)

Evaluates multi-choice predictions with type-based accuracy statistics. Supports both `model_prediction`/`ground_truth` and `model_output`/`answer` field formats.

#### Example

```bash
export CUDA_VISIBLE_DEVICES=0

INPUT_DIR="../RESULT/refer/qwen3vl_8b/gar_image_mc_qa/inst_it_image/04_rew_n1024/01_baseline/02_judged"
PATTERN="results_layer_*_multiplier_*_behavior_refer_type_gar_image_mc_qa*"
LOG_DIR="${INPUT_DIR}/evals_gar_vqa"
SUMMARY_OUT="${LOG_DIR}/summary_gar_vqa.csv"

mkdir -p "${LOG_DIR}"

python evaluate_gar.py \
    --eval_type vqa \
    --input_dir "${INPUT_DIR}" \
    --pattern "${PATTERN}" \
    --log_dir "${LOG_DIR}" \
    --summary_out "${SUMMARY_OUT}"
```

#### Output

- Per-file metrics: `{filename}.metrics.json` in `log_dir`
- Summary CSV with columns:
  - `file`, `layer`, `multiplier`, `behavior`, `type`, `model_name`, `model_size`
  - `accuracy`, `valid_num`, `correct_num`, `total_num`
  - `acc_Color`, `acc_Shape`, `acc_Texture`, `acc_Material`, `acc_Position`, `acc_Non-Entity`, `acc_Relation`

### GAR-Simple

Evaluates simple format predictions using vLLM. Requires vLLM service running.

#### Example

```bash
export CUDA_VISIBLE_DEVICES=0

MODEL_DIR="qwen3vl_8b"
INPUT_DIR="../RESULT/refer/qwen3vl_8b/gar_image_simple_oe_qa/inst_it_image/04_rew_n1024/01_baseline/02_judged"
PATTERN="results_layer_*_multiplier_*_behavior_refer_type_gar_image_simple_oe_qa_model_name_*_model_size_*.json"

# vLLM configuration
BASE_URL="http://localhost:8001/v1"
MODEL_NAME="Qwen/Qwen2.5-72B-Instruct-AWQ"
NUM_PROC=8

LOG_DIR="${INPUT_DIR}/evals_gar_simple"
SUMMARY_OUT="${LOG_DIR}/summary_gar_simple.csv"

mkdir -p "${LOG_DIR}"

python evaluate_gar.py \
    --eval_type simple \
    --input_dir "${INPUT_DIR}" \
    --pattern "${PATTERN}" \
    --log_dir "${LOG_DIR}" \
    --summary_out "${SUMMARY_OUT}" \
    --base_url "${BASE_URL}" \
    --model_name "${MODEL_NAME}" \
    --num_process "${NUM_PROC}"
```

#### Output

- Per-file metrics: `{filename}.metrics.json` in `log_dir`
- Summary CSV with columns:
  - `file`, `layer`, `multiplier`, `behavior`, `type`, `model_name`, `model_size`
  - `accuracy`, `valid_num`, `true_num`, `total_num`

### GAR-Detailed

Evaluates detailed format predictions using vLLM. Requires vLLM service running.

#### Example

```bash
export CUDA_VISIBLE_DEVICES=0

MODEL_DIR="qwen3vl_8b"
INPUT_DIR="../RESULT/refer/qwen3vl_8b/gar_image_detail_oe_qa/inst_it_image/04_rew_n1024/01_baseline/02_judged"
PATTERN="results_layer_*_multiplier_*_behavior_refer_type_gar_image_detail_oe_qa_model_name_*_model_size_*.json"

# vLLM configuration
BASE_URL="http://localhost:8001/v1"
MODEL_NAME="Qwen/Qwen2.5-72B-Instruct-AWQ"
NUM_PROC=8

LOG_DIR="${INPUT_DIR}/evals_gar_detailed"
SUMMARY_OUT="${LOG_DIR}/summary_gar_detailed.csv"

mkdir -p "${LOG_DIR}"

python evaluate_gar.py \
    --eval_type detailed \
    --input_dir "${INPUT_DIR}" \
    --pattern "${PATTERN}" \
    --log_dir "${LOG_DIR}" \
    --summary_out "${SUMMARY_OUT}" \
    --base_url "${BASE_URL}" \
    --model_name "${MODEL_NAME}" \
    --num_process "${NUM_PROC}"
```

#### Output

- Per-file metrics: `{filename}.metrics.json` in `log_dir`
- Summary CSV with columns:
  - `file`, `layer`, `multiplier`, `behavior`, `type`, `model_name`, `model_size`
  - `accuracy`, `valid_num`, `true_num`, `total_num`

## Layer Range Filtering

You can filter files by layer range to evaluate only specific layers:

```bash
python evaluate_gar.py \
    --eval_type detailed \
    --input_dir "${INPUT_DIR}" \
    --log_dir "${LOG_DIR}" \
    --layer_range 0 35 \
    --base_url "${BASE_URL}" \
    --model_name "${MODEL_NAME}"
```

This will only evaluate files with layer numbers between 0 and 35 (inclusive).

## File Pattern Auto-Detection

If `--pattern` is not provided, the script automatically detects patterns based on `--eval_type`:

- `detailed`: `results_layer_*_multiplier_*_behavior_refer_type_gar_image_detail_oe_qa_model_name_*_model_size_*.json`
- `simple`: `results_layer_*_multiplier_*_behavior_refer_type_gar_image_simple_oe_qa_model_name_*_model_size_*.json`
- `vqa`: `results_layer_*_multiplier_*_behavior_refer_type_gar_image_mc_qa*`

## Output Format

### Per-File Metrics JSON

Each prediction file generates a corresponding `.metrics.json` file containing:

```json
{
    "accuracy": 85.5,
    "total_num": 100,
    "valid_num": 98,
    "correct_num": 84,
    "timestamp": "2026-01-23 10:30:00",
    "evaluator_model": "Qwen/Qwen2.5-72B-Instruct-AWQ",
    "evaluator_base_url": "http://localhost:8001/v1",
    "eval_prompt": "...",
    "model_predictons": [...]
}
```

### Summary CSV

The summary CSV aggregates results from all evaluated files, with metadata extracted from filenames (layer, multiplier, behavior, type, model_name, model_size).

## Notes

- For GAR-VQA, the tool supports both field name formats: `model_prediction`/`ground_truth` and `model_output`/`answer`
- vLLM service must be running for GAR-Simple and GAR-Detailed evaluations
- The script uses parallel processing for faster evaluation (configurable via `--num_process`)
- Metadata (layer, multiplier, etc.) is automatically extracted from filenames using regex patterns

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
