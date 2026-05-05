# Inst-It-Bench Evaluation Tool

Unified evaluation tool for Inst-It-Bench formats: Image MC/OE and Video MC/OE.

## Setup

Ensure you have the required dependencies installed:

```bash
pip install openai tqdm pandas huggingface_hub
```

For Open-Ended (OE) evaluations, you need a vLLM service running. Start it with:

```bash
# Example: Start vLLM service on port 8001
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-72B-Instruct-AWQ \
    --port 8001
```

## Evaluation Types

The tool supports four evaluation types:

- **Image MC**: Image multi-choice question answering
- **Image OE**: Image open-ended question answering (requires vLLM judge)
- **Video MC**: Video multi-choice question answering
- **Video OE**: Video open-ended question answering (requires vLLM judge)

## Usage

### Qwen3-VL

#### Inst-It-Image-MC

```bash
cd /home/dengjie/liuh0115/caa-gar
export CUDA_VISIBLE_DEVICES=0

INPUT_DIR="../MSteer--RESULT/refer/qwen3vl_8b/inst_it_image_mc_qa/inst_it_image/04_rew_n1024/01_baseline/01_raw"
OUTPUT_DIR="../MSteer--RESULT/refer/qwen3vl_8b/inst_it_image_mc_qa/inst_it_image/04_rew_n1024/01_baseline/02_judged"
SUMMARY_OUT="../MSteer--RESULT/refer/qwen3vl_8b/inst_it_image_mc_qa/inst_it_image/04_rew_n1024/01_baseline/summary.csv"

python evaluate_inst_it.py \
    --eval_type image_mc \
    --input_dir "${INPUT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --summary_out "${SUMMARY_OUT}"
```

#### Inst-It-Image-OE

```bash
cd /home/dengjie/liuh0115/caa-gar
export CUDA_VISIBLE_DEVICES=0

INPUT_DIR="../MSteer--RESULT/refer/qwen3vl_8b/inst_it_image_oe_qa/inst_it_image/04_rew_n1024/01_baseline/01_raw"
BASE_URL="http://localhost:8001/v1"
MODEL_NAME="Qwen/Qwen2.5-72B-Instruct-AWQ"
NUM_WORKERS=8

python evaluate_inst_it.py \
    --eval_type image_oe \
    --input_dir "${INPUT_DIR}" \
    --base_url "${BASE_URL}" \
    --model_name "${MODEL_NAME}" \
    --num_workers "${NUM_WORKERS}"
```

#### Inst-It-Video-MC

```bash
cd /home/dengjie/liuh0115/caa-gar
export CUDA_VISIBLE_DEVICES=0

INPUT_DIR="../MSteer--RESULT/refer/qwen3vl_8b/inst_it_video_mc_qa/inst_it_image/04_rew_n1024/01_baseline/01_raw"
OUTPUT_DIR="../MSteer--RESULT/refer/qwen3vl_8b/inst_it_video_mc_qa/inst_it_image/04_rew_n1024/01_baseline/02_judged"
SUMMARY_OUT="../MSteer--RESULT/refer/qwen3vl_8b/inst_it_video_mc_qa/inst_it_image/04_rew_n1024/01_baseline/summary.csv"

python evaluate_inst_it.py \
    --eval_type video_mc \
    --input_dir "${INPUT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --summary_out "${SUMMARY_OUT}"
```

#### Inst-It-Video-OE

```bash
cd /home/dengjie/liuh0115/caa-gar
export CUDA_VISIBLE_DEVICES=0

INPUT_DIR="../MSteer--RESULT/refer/qwen3vl_8b/inst_it_video_oe_qa/inst_it_image/04_rew_n1024/01_baseline/01_raw"
BASE_URL="http://localhost:8001/v1"
MODEL_NAME="Qwen/Qwen2.5-72B-Instruct-AWQ"
NUM_WORKERS=8

python evaluate_inst_it.py \
    --eval_type video_oe \
    --input_dir "${INPUT_DIR}" \
    --base_url "${BASE_URL}" \
    --model_name "${MODEL_NAME}" \
    --num_workers "${NUM_WORKERS}"
```

### InternVL-3.5

#### Inst-It-Image-MC

```bash
cd /home/dengjie/liuh0115/caa-gar
export CUDA_VISIBLE_DEVICES=0

INPUT_DIR="../MSteer--RESULT/refer/internvl3_5_8b/inst_it_image_mc_qa/inst_it_image/04_rew_n1024/01_baseline/01_raw"
OUTPUT_DIR="../MSteer--RESULT/refer/internvl3_5_8b/inst_it_image_mc_qa/inst_it_image/04_rew_n1024/01_baseline/02_judged"
SUMMARY_OUT="../MSteer--RESULT/refer/internvl3_5_8b/inst_it_image_mc_qa/inst_it_image/04_rew_n1024/01_baseline/summary.csv"

python evaluate_inst_it.py \
    --eval_type image_mc \
    --input_dir "${INPUT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --summary_out "${SUMMARY_OUT}"
```

#### Inst-It-Image-OE

```bash
cd /home/dengjie/liuh0115/caa-gar
export CUDA_VISIBLE_DEVICES=0

INPUT_DIR="../MSteer--RESULT/refer/internvl3_5_8b/inst_it_image_oe_qa/inst_it_image/04_rew_n1024/01_baseline/01_raw"
BASE_URL="http://localhost:8001/v1"
MODEL_NAME="Qwen/Qwen2.5-72B-Instruct-AWQ"
NUM_WORKERS=8

python evaluate_inst_it.py \
    --eval_type image_oe \
    --input_dir "${INPUT_DIR}" \
    --base_url "${BASE_URL}" \
    --model_name "${MODEL_NAME}" \
    --num_workers "${NUM_WORKERS}"
```

#### Inst-It-Video-MC

```bash
cd /home/dengjie/liuh0115/caa-gar
export CUDA_VISIBLE_DEVICES=0

INPUT_DIR="../MSteer--RESULT/refer/internvl3_5_8b/inst_it_video_mc_qa/inst_it_image/04_rew_n1024/01_baseline/01_raw"
OUTPUT_DIR="../MSteer--RESULT/refer/internvl3_5_8b/inst_it_video_mc_qa/inst_it_image/04_rew_n1024/01_baseline/02_judged"
SUMMARY_OUT="../MSteer--RESULT/refer/internvl3_5_8b/inst_it_video_mc_qa/inst_it_image/04_rew_n1024/01_baseline/summary.csv"

python evaluate_inst_it.py \
    --eval_type video_mc \
    --input_dir "${INPUT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --summary_out "${SUMMARY_OUT}"
```

#### Inst-It-Video-OE

```bash
cd /home/dengjie/liuh0115/caa-gar
export CUDA_VISIBLE_DEVICES=0

INPUT_DIR="../MSteer--RESULT/refer/internvl3_5_8b/inst_it_video_oe_qa/inst_it_image/04_rew_n1024/01_baseline/01_raw"
BASE_URL="http://localhost:8001/v1"
MODEL_NAME="Qwen/Qwen2.5-72B-Instruct-AWQ"
NUM_WORKERS=8

python evaluate_inst_it.py \
    --eval_type video_oe \
    --input_dir "${INPUT_DIR}" \
    --base_url "${BASE_URL}" \
    --model_name "${MODEL_NAME}" \
    --num_workers "${NUM_WORKERS}"
```

## Arguments

### Required Arguments

- `--eval_type`: Evaluation type - `image_mc`, `image_oe`, `video_mc`, or `video_oe`
- `--input_dir`: Input directory containing JSON files (usually `01_raw`)

### Optional Arguments

- `--output_dir`: Output directory (default: `02_judged` for OE, same as input for MC)
- `--pattern`: Glob pattern to match files (auto-detected if not provided)
- `--summary_out`: CSV file path to save summary
- `--base_url`: vLLM OpenAI API endpoint (default: `http://localhost:8001/v1`, required for OE)
- `--model_name`: Judge model name for OE (default: `Qwen/Qwen2.5-72B-Instruct-AWQ`)
- `--num_workers`: Number of parallel workers for OE (default: 8)
- `--overwrite`: Overwrite existing scores
- `--layer_range`: Only evaluate layers in this range (e.g., `0 35`)
- `--no_load_gt`: Do not load ground truth from Inst-It-Bench dataset

## File Pattern Auto-Detection

If `--pattern` is not provided, the script automatically detects patterns based on `--eval_type`:

- `image_mc`: `results_layer_*_multiplier_*_behavior_refer_type_inst_it_image_mc_qa_model_name_*_model_size_*.json`
- `image_oe`: `results_layer_*_multiplier_*_behavior_refer_type_inst_it_image_oe_qa_model_name_*_model_size_*.json`
- `video_mc`: `results_layer_*_multiplier_*_behavior_refer_type_inst_it_video_mc_qa_model_name_*_model_size_*.json`
- `video_oe`: `results_layer_*_multiplier_*_behavior_refer_type_inst_it_video_oe_qa_model_name_*_model_size_*.json`

## Output Format

### Multi-Choice (MC) Output

- Per-file metrics: `{filename}.json` in output directory
- Summary CSV with columns:
  - `file`, `layer`, `multiplier`, `behavior`, `type`, `model_name`, `model_size`
  - `accuracy`, `valid_num`, `correct_num`, `total_num`

### Open-Ended (OE) Output

- Per-file metrics: `{filename}_evaluated.json` in `02_judged` directory
- Summary CSV with columns:
  - `file`, `layer`, `multiplier`, `behavior`, `type`, `model_name`, `model_size`
  - `accuracy`, `valid_num`, `total_num`, `avg_score`

## Notes

- MC evaluations automatically load ground truth from Inst-It-Bench dataset
- OE evaluations require a vLLM service running for scoring
- The tool automatically extracts answers from model outputs for MC evaluations
- For OE evaluations, the tool uses vLLM judge to score responses on a 0-1 scale
- Metadata (layer, multiplier, etc.) is automatically extracted from filenames
- Output directories are automatically set: MC outputs to same directory, OE outputs to `02_judged`
- If `--summary_out` is not specified, `summary.csv` is automatically saved in the parent directory of `OUTPUT_DIR` (same level as `OUTPUT_DIR`)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
