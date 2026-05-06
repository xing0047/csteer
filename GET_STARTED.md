# Getting Started

## Data

Make sure the data for building vectors is prepared as below in `datasets`. We use `INST-IT-Image` and `INST-IT-Video` for our experiments. 

```text 
datasets
├── Inst-It-Dataset
│   ├── inst_it_dataset_image_51k.json
│   ├── inst_it_dataset_video_21k.json
│   ├── images_vpt
│   ├── images_raw
│   ├── videos_vpt
│   └── videos_raw
```

## Serve

Deploy `Qwen/Qwen2.5-72B-Instruct-AWQ` on additional GPUs as an OpenAI-compatible service.
This service is used for both:

- **score** in `rollout_with_score.py`
- **rewrite** in `eval_lhy/rewrite_rollouts.py`

Example (use extra GPUs `4,5,6,7`):

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-72B-Instruct-AWQ \
    --served-model-name Qwen/Qwen2.5-72B-Instruct-AWQ \
    --tensor-parallel-size 4 \
    --port 8001 \
    --dtype auto
```

Then set:

- `--judge_base_url http://localhost:8001/v1`
- `--judge_model_name Qwen/Qwen2.5-72B-Instruct-AWQ`


## Rollout 

Run `rollout_with_score.py` to generate multiple rollouts and scores each rollout with the judge model.

```bash
python rollout_with_score.py \
    --model_name qwen3vl \
    --model_size 8b \
    --data_type image \
    --data_path datasets/Inst-It-Dataset/inst_it_dataset_image_51k.json \
    --media_root datasets/Inst-It-Dataset \
    --n_samples 1024 \
    --num_rollouts 8 \
    --judge_base_url http://localhost:8001/v1 \
    --judge_model_name Qwen/Qwen2.5-72B-Instruct-AWQ \
    --output_dir image_rollout_exp \
    --verbose
```

## Outputs

Results are saved to:

- `ROLLOUT_RESULTS/<output_dir>/judge_results.json`
- `ROLLOUT_RESULTS/<output_dir>/judge_results.csv`

