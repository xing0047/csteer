model_name=${1:-"internvl3"}
model_size=${2:-"2b"}
n_layers=${3:-28}
type=${4:-"inst_it_video_oe_qa"}

# 1. Generate steering pairs
# python generate.py \
#     --model_name ${model_name} \
#     --model_size ${model_size}

# 2. Generate steering vectors for layers of the model for a certain behavior
# python generate_vectors.py \
#     --layers $(seq 0 $((n_layers-1))) \
#     --save_activations \
#     --model_name ${model_name} \
#     --model_size ${model_size} \
#     --behaviors refer

# 3. Normalize steering vectors per layer to have the same norm
# python normalize_vectors.py \
#     --model_name ${model_name} \
#     --model_size ${model_size} \
#     --n_layers ${n_layers}

# 4. Evaluate model on open-ended image referential, including baseline (with multipliers set to 0), and CAA across layers
python prompting_with_steering.py \
    --behaviors refer \
    --type ${type} \
    --model_name ${model_name} \
    --model_size ${model_size} \
    --layers 0 \
    --multipliers 0 
# python prompting_with_steering.py \
#     --behaviors refer \
#     --type ${type} \
#     --model_name ${model_name} \
#     --model_size ${model_size} \
#     --layers $(seq 0 $((n_layers-1))) \
#     --multipliers 1 

