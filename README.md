# Steering Llama 2 with Contrastive Activation Addition

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Then create a `.env` file with the following variables (see `.env.example`):

```
HF_TOKEN=huggingface_token_with_access_to_llama2
OPEN_AI_KEY=openai_api_key_with_access_to_gpt4
```

## Datasets for Generating Steering Vectors.

### Inst-It-Image (`inst_it_image`)

### Inst-It-Video (`inst_it_video`)

## Final dataset sizes

```
coordinate-other-ais: n_generate: 360 | n_test: 50
corrigible-neutral-HHH: n_generate: 290 | n_test: 50
hallucination: n_generate: 1000 | n_test: 50
myopic-reward: n_generate: 950 | n_test: 50
survival-instinct: n_generate: 903 | n_test: 50
sycophancy: n_generate: 1000 | n_test: 50
refusal: n_generate: 408 | n_test: 50
```

## Evaluation

### Qwen3-VL

#### Inst-It-Image-MC

```bash
cd /home/xingy/mosteer/caa
conda activate xy_refer_caa
MODEL_NAME=qwen3vl
MODEL_SIZE=2b
MODEL_LAYERS=$(seq 0 27)
VECTOR_DATA=inst_it_image
TEST_DATA=inst_it_image_mc_qa
N_PAIRS=1024
EXP_NAME=inst_it_image_mo_dev_01
HF_USER=xing0047
export CUDA_VISIBLE_DEVICES=2
```

#### Inst-It-Image-OE

```bash
cd /home/xingy/mosteer/caa
conda activate xy_refer_caa
MODEL_NAME=qwen3vl
MODEL_SIZE=2b
MODEL_LAYERS=$(seq 0 27)
VECTOR_DATA=inst_it_image
TEST_DATA=inst_it_image_oe_qa
N_PAIRS=1024
EXP_NAME=inst_it_image_mo_dev_01
HF_USER=xing0047
export CUDA_VISIBLE_DEVICES=2
```

#### Inst-It-Video-MC

```bash
cd /home/xingy/mosteer/caa
conda activate xy_refer_caa
MODEL_NAME=qwen3vl
MODEL_SIZE=2b
MODEL_LAYERS=$(seq 0 27)
VECTOR_DATA=inst_it_video
TEST_DATA=inst_it_video_mc_qa
N_PAIRS=1024
EXP_NAME=inst_it_video_mo_dev_01
HF_USER=xing0047
export CUDA_VISIBLE_DEVICES=2
```

#### Inst-It-Video-OE

```bash
cd /home/xingy/mosteer/caa
conda activate xy_refer_caa
MODEL_NAME=qwen3vl
MODEL_SIZE=2b
MODEL_LAYERS=$(seq 0 27)
VECTOR_DATA=inst_it_video
TEST_DATA=inst_it_video_oe_qa
N_PAIRS=1024
EXP_NAME=inst_it_video_mo_dev_01
HF_USER=xing0047
export CUDA_VISIBLE_DEVICES=2
```




### InternVL-3.5

#### Inst-It-Image-MC

```bash
cd /home/xingy/mosteer/caa
conda activate xy_refer_caa
MODEL_NAME=internvl3_5
MODEL_SIZE=2b
MODEL_LAYERS=$(seq 0 27)
VECTOR_DATA=inst_it_image
TEST_DATA=inst_it_image_mc_qa
N_PAIRS=1024
EXP_NAME=inst_it_image_mo_dev_01
HF_USER=xing0047
export CUDA_VISIBLE_DEVICES=2
```

#### Inst-It-Image-OE

```bash
cd /home/xingy/mosteer/caa
conda activate xy_refer_caa
MODEL_NAME=internvl3_5
MODEL_SIZE=2b
MODEL_LAYERS=$(seq 0 27)
VECTOR_DATA=inst_it_image
TEST_DATA=inst_it_image_oe_qa
N_PAIRS=1024
EXP_NAME=inst_it_image_mo_dev_01
HF_USER=xing0047
export CUDA_VISIBLE_DEVICES=2
```

#### Inst-It-Video-MC

```bash
cd /home/xingy/mosteer/caa
conda activate xy_refer_caa
MODEL_NAME=internvl3_5
MODEL_SIZE=2b
MODEL_LAYERS=$(seq 0 27)
VECTOR_DATA=inst_it_video
TEST_DATA=inst_it_video_mc_qa
N_PAIRS=1024
EXP_NAME=inst_it_video_mo_dev_01
HF_USER=xing0047
export CUDA_VISIBLE_DEVICES=2
```

#### Inst-It-Video-OE

```bash
cd /home/xingy/mosteer/caa
conda activate xy_refer_caa
MODEL_NAME=internvl3_5
MODEL_SIZE=2b
MODEL_LAYERS=$(seq 0 27)
VECTOR_DATA=inst_it_video
TEST_DATA=inst_it_video_oe_qa
N_PAIRS=1024
EXP_NAME=inst_it_video_mo_dev_01
HF_USER=xing0047
export CUDA_VISIBLE_DEVICES=2
```

### Command

```bash
python generate.py \
    --model_name ${MODEL_NAME} \
    --model_size ${MODEL_SIZE} \
    --data ${VECTOR_DATA} \
    --n_pairs ${N_PAIRS} \
    --output_dir ../PAIRS/${EXP_NAME} \
    --hf_user ${HF_USER} \
    --use_flash_attn \
    --verbose

python generate_vector.py \
    --model_name ${MODEL_NAME} \
    --model_size ${MODEL_SIZE} \
    --layers ${MODEL_LAYERS} \
    --behaviors refer \
    --behavior_paths ../PAIRS/${EXP_NAME}/${HF_USER}_${MODEL_NAME}_${MODEL_SIZE}_${VECTOR_DATA}_n${N_PAIRS} \
    --output_dir ${EXP_NAME} \
    --use_flash_attn \
    --verbose

python normalize_vector.py \
    --model_name ${MODEL_NAME} \
    --model_size ${MODEL_SIZE} \
    --vector_suffix ${EXP_NAME} \
    --layers ${MODEL_LAYERS}

python prompting_with_steering.py \
    --model_name ${MODEL_NAME} \
    --model_size ${MODEL_SIZE} \
    --use_flash_attn \
    --behaviors refer \
    --output_dir ${EXP_NAME} \
    --type ${TEST_DATA} \
    --layers 0 \
    --multipliers 0.0 \
    --verbose

python prompting_with_steering.py \
    --model_name ${MODEL_NAME} \
    --model_size ${MODEL_SIZE} \
    --use_flash_attn \
    --behaviors refer \
    --output_dir ${EXP_NAME} \
    --type ${TEST_DATA} \
    --layers ${MODEL_LAYERS} \
    --multipliers 1.0 \
    --verbose
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
