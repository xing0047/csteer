# Rollout-Rewrite Workflow Usage Guide

This guide demonstrates how to use the three-step rollout-rewrite workflow to generate steering vectors from rollouts and their rewritten versions.

## Workflow Overview

The rollout-rewrite workflow consists of three steps:

1. **Generate Rollouts** (`generate_rollout.py`): Generate multiple rollouts for each sample and score them using a judge model
2. **Rewrite Rollouts** (`rewrite_rollouts.py`): Rewrite low-score rollouts to improve their accuracy
3. **Generate Vectors** (`generate_vector_rewrite.py`): Generate steering vectors by comparing rewritten (positive) and original (negative) rollouts

## Supported Models

- `internvl3`
- `internvl3_5`
- `qwen3vl`

## Supported Data Types

- `image`: Image mode
- `video`: Video mode

## Examples

### InternVL-3.5

#### Inst-It-Image

```bash
cd /path/to/caa
conda activate your_env
MODEL_NAME=internvl3_5
MODEL_SIZE=8b
MODEL_LAYERS=$(seq 0 35)
DATA_TYPE=image
DATA_PATH=../DATA/Inst-It-Dataset/inst_it_dataset_image_51k.json
MEDIA_ROOT=../DATA/Inst-It-Dataset
N_SAMPLES=1024
NUM_ROLLOUTS=8
ROLLOUT_TEMP=0.6
SCORE_THRESHOLD=0.6
JUDGE_BASE_URL=http://localhost:8001/v1
JUDGE_MODEL_NAME=Qwen/Qwen2.5-72B-Instruct-AWQ
OUTPUT_DIR=image_rollout_rewrite_exp
export CUDA_VISIBLE_DEVICES=0
```

#### Inst-It-Video

```bash
cd /path/to/caa
conda activate your_env
MODEL_NAME=internvl3_5
MODEL_SIZE=8b
MODEL_LAYERS=$(seq 0 35)
DATA_TYPE=video
DATA_PATH=../DATA/Inst-It-Dataset/inst_it_dataset_video_21k.json
MEDIA_ROOT=../DATA/Inst-It-Dataset
N_SAMPLES=1024
NUM_ROLLOUTS=8
ROLLOUT_TEMP=0.6
SCORE_THRESHOLD=0.6
JUDGE_BASE_URL=http://localhost:8001/v1
JUDGE_MODEL_NAME=Qwen/Qwen2.5-72B-Instruct-AWQ
OUTPUT_DIR=video_rollout_rewrite_exp
export CUDA_VISIBLE_DEVICES=0
```

### Qwen3-VL

#### Inst-It-Image

```bash
cd /path/to/caa
conda activate your_env
MODEL_NAME=qwen3vl
MODEL_SIZE=8b
MODEL_LAYERS=$(seq 0 35)
DATA_TYPE=image
DATA_PATH=../DATA/Inst-It-Dataset/inst_it_dataset_image_51k.json
MEDIA_ROOT=../DATA/Inst-It-Dataset
N_SAMPLES=1024
NUM_ROLLOUTS=8
ROLLOUT_TEMP=0.6
SCORE_THRESHOLD=0.6
JUDGE_BASE_URL=http://localhost:8001/v1
JUDGE_MODEL_NAME=Qwen/Qwen2.5-72B-Instruct-AWQ
OUTPUT_DIR=image_rollout_rewrite_exp
export CUDA_VISIBLE_DEVICES=0
```

#### Inst-It-Video

```bash
cd /path/to/caa
conda activate your_env
MODEL_NAME=qwen3vl
MODEL_SIZE=8b
MODEL_LAYERS=$(seq 0 35)
DATA_TYPE=video
DATA_PATH=../DATA/Inst-It-Dataset/inst_it_dataset_video_21k.json
MEDIA_ROOT=../DATA/Inst-It-Dataset
N_SAMPLES=1024
NUM_ROLLOUTS=8
ROLLOUT_TEMP=0.6
SCORE_THRESHOLD=0.6
JUDGE_BASE_URL=http://localhost:8001/v1
JUDGE_MODEL_NAME=Qwen/Qwen2.5-72B-Instruct-AWQ
OUTPUT_DIR=video_rollout_rewrite_exp
export CUDA_VISIBLE_DEVICES=0
```

## Commands

After setting the environment variables above, run the following three commands in sequence:

### Step 1: Generate Rollouts

Generate multiple rollouts for each sample and score them using a judge model.

```bash
python generate_rollout.py \
    --model_name ${MODEL_NAME} \
    --model_size ${MODEL_SIZE} \
    --data_type ${DATA_TYPE} \
    --data_path ${DATA_PATH} \
    --media_root ${MEDIA_ROOT} \
    --n_samples ${N_SAMPLES} \
    --num_rollouts ${NUM_ROLLOUTS} \
    --rollout_temperature ${ROLLOUT_TEMP} \
    --judge_base_url ${JUDGE_BASE_URL} \
    --judge_model_name ${JUDGE_MODEL_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --verbose
```

**Output**: `ROLLOUT_RESULTS/${OUTPUT_DIR}/judge_results.json`

This file contains:
- Original rollouts with their scores
- Sample metadata (media paths, ground truth, etc.)

### Step 2: Rewrite Low-Score Rollouts

Rewrite rollouts with scores below the threshold to improve their accuracy.

```bash
python rewrite_rollouts.py \
    --data_type ${DATA_TYPE} \
    --input_json ROLLOUT_RESULTS/${OUTPUT_DIR}/judge_results.json \
    --output_json ROLLOUT_RESULTS/${OUTPUT_DIR}/rewritten_rollouts.json \
    --judge_base_url ${JUDGE_BASE_URL} \
    --judge_model_name ${JUDGE_MODEL_NAME} \
    --score_threshold ${SCORE_THRESHOLD} \
    --verbose
```

**Output**: `ROLLOUT_RESULTS/${OUTPUT_DIR}/rewritten_rollouts.json`

This file contains:
- Rewritten versions of low-score rollouts
- Matched with original rollouts by sample_id and rollout_id

### Step 3: Generate Steering Vectors

Generate steering vectors by comparing rewritten (positive) and original (negative) rollouts.

```bash
python generate_vector_rewrite.py \
    --model_name ${MODEL_NAME} \
    --model_size ${MODEL_SIZE} \
    --data_type ${DATA_TYPE} \
    --layers ${MODEL_LAYERS} \
    --judge_results_json ROLLOUT_RESULTS/${OUTPUT_DIR}/judge_results.json \
    --rewritten_rollouts_json ROLLOUT_RESULTS/${OUTPUT_DIR}/rewritten_rollouts.json \
    --data_path ${DATA_PATH} \
    --media_root ${MEDIA_ROOT} \
    --n_samples ${N_SAMPLES} \
    --score_threshold ${SCORE_THRESHOLD} \
    --output_dir ${OUTPUT_DIR} \
    --use_flash_attn \
    --verbose
```

**Output**: Steering vectors saved to `VECTORS/refer_vpt/${MODEL_NAME}_${MODEL_SIZE}_${OUTPUT_DIR}/layer_*.pt`

**Note on `--n_samples`**: This parameter specifies the number of **GT samples** (ground truth samples) to use, not the number of rollout pairs. The script will:
- Select the first `${N_SAMPLES}` GT samples that have corresponding entries in both `judge_results.json` and `rewritten_rollouts.json`
- Include **all** qualifying rollouts (score ≤ `${SCORE_THRESHOLD}`) from these GT samples
- The total number of rollout pairs may be larger than `${N_SAMPLES}` if each GT has multiple qualifying rollouts

## Notes

1. **Judge Model Service**: Make sure the judge model service is running at `${JUDGE_BASE_URL}` before running the scripts. The judge model is used for both scoring rollouts (Step 1) and rewriting them (Step 2).

2. **GPU Memory**: For large models or video data, you may need to adjust GPU memory settings or use tensor parallelism.

3. **Score Threshold**: Rollouts with scores ≤ `${SCORE_THRESHOLD}` will be rewritten. The default is 0.6.

4. **N_SAMPLES Parameter**: 
   - In **Step 1** (`generate_rollout.py`): `--n_samples` limits the number of GT samples to process
   - In **Step 3** (`generate_vector_rewrite.py`): `--n_samples` specifies the number of GT samples to use, and **all** qualifying rollouts from these samples will be included
   - Example: If `N_SAMPLES=1024` and each GT has an average of 5 qualifying rollouts, Step 3 will use approximately 5120 rollout pairs

5. **Layer Range**: Adjust `${MODEL_LAYERS}` based on your model:
   - InternVL-3.5 (8b): typically `$(seq 0 35)`
   - Qwen3-VL (8b): typically `$(seq 0 35)`

6. **Output Structure**: 
   - Rollout results: `ROLLOUT_RESULTS/${OUTPUT_DIR}/`
   - Steering vectors: `VECTORS/refer_vpt/${MODEL_NAME}_${MODEL_SIZE}_${OUTPUT_DIR}/`
