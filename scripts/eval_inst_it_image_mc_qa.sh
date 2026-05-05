model_name=${1:-"internvl3_5"}
model_size=${2:-"8b"}
n_layers=${3:-28}
type=${4:-"inst_it_image_mc_qa"}

export CUDA_VISIBLE_DEVICES=0,1

# 1. Generate steering pairs
python generate.py \
    --model_name ${model_name} \
    --model_size ${model_size} \
    --n_pairs 1024 \
    --hf_user xing0047

# python generate_vectors_inst.py \
#     --layers $(seq 0 $((n_layers-1))) \
#     --model_name ${model_name} \
#     --model_size ${model_size} \
#     --behaviors refer

# python normalize_vectors.py \
#     --model_name ${model_name} \
#     --model_size ${model_size} \
#     --n_layers ${n_layers}

# python prompting_with_steering.py \
#     --behaviors refer \
#     --type ${type} \
#     --model_name ${model_name} \
#     --model_size ${model_size} \
#     --layers 0 \
#     --multipliers 0 

# python prompting_with_steering.py \
#     --behaviors refer \
#     --type ${type} \
#     --model_name ${model_name} \
#     --model_size ${model_size} \
#     --layers $(seq 0 $((n_layer