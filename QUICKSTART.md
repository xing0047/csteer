# Getting Started

## Data

Make sure the data for building vectors is prepared as mentioned in DATA.md. We use `INST-IT-Image` for the example. 

```text 
datasets
├── Inst-It-Dataset
│   ├── inst_it_dataset_image_51k.json
│   ├── images_vpt
│   └── images_raw
```

## Serve

Deploy `Qwen/Qwen2.5-72B-Instruct-AWQ` on additional GPUs as an OpenAI-compatible service.
This service is used for both:

- **score** in `rollout_with_score.py`
- **rewrite** in `rewrite.py`

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-72B-Instruct-AWQ \
    --served-model-name Qwen/Qwen2.5-72B-Instruct-AWQ \
    --tensor-parallel-size 4 \
    --port 8000 \
    --dtype auto
```

Then set:

- `--judge_base_url http://localhost:8000/v1`
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
    --judge_base_url http://localhost:8000/v1 \
    --judge_model_name Qwen/Qwen2.5-72B-Instruct-AWQ \
    --output_dir image_rollout_exp \
    --verbose
```

## Rewrite

Run `rewrite.py` to rewrite low-score rollouts from Step 1.

```bash
python rewrite.py \
    --data_type image \
    --input_json ROLLOUT_RESULTS/image_rollout_exp/judge_results.json \
    --output_json ROLLOUT_RESULTS/image_rollout_exp/rewritten_rollouts.json \
    --judge_base_url http://localhost:8000/v1 \
    --judge_model_name Qwen/Qwen2.5-72B-Instruct-AWQ \
    --score_threshold 0.6 \
    --verbose
```

## Gen Vector

Run `gen_vector.py` to build steering vectors from rewritten (positive) vs original rollout (negative) pairs.

```bash
python gen_vector.py \
    --model_name qwen3vl \
    --model_size 8b \
    --data_type image \
    --layers $(seq 0 35) \
    --judge_results_json ROLLOUT_RESULTS/image_rollout_exp/judge_results.json \
    --rewritten_rollouts_json ROLLOUT_RESULTS/image_rollout_exp/rewritten_rollouts.json \
    --data_path datasets/Inst-It-Dataset/inst_it_dataset_image_51k.json \
    --media_root datasets/Inst-It-Dataset \
    --n_samples 1024 \
    --score_threshold 0.6 \
    --output_dir refer_rewrite_image_exp \
    --use_flash_attn \
    --verbose
```

## Norm Vector

Run `norm_vector.py` to normalize vector norms across behaviors/layers.

```bash
python norm_vector.py \
    --model_name qwen3vl \
    --vector_suffix refer_rewrite_image_exp \
    --layers $(seq 0 35)
```

## Prompt With Steering

Run `prompting_with_steering.py` (your "prompt_with_steering" step) to evaluate steering at inference time.

```bash
python prompting_with_steering.py \
    --behaviors refer \
    --type inst_it_image_oe_qa \
    --model_name qwen3vl \
    --model_size 8b \
    --vector_dir refer_rewrite_image_exp \
    --output_dir refer_rewrite_image_exp \
    --layers $(seq 0 35) \
    --multipliers 1 \
    --use_flash_attn \
    --verbose
```

## Judge Logic

Use `eval_outputs.py` to run the final judge/evaluation on generated outputs.

```bash
python eval_outputs.py inst_it \
    --mode image_oe \
    --input <path_to_results_json_from_prompting_with_steering> \
    --base_url http://localhost:8000/v1 \
    --model_name Qwen/Qwen2.5-72B-Instruct-AWQ
```


