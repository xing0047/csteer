model_name=${1:-"internvl3"}
model_size=${2:-"2b"}
n_layers=${3:-28}
type=${4:-"inst_it_video_mc_qa"}

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

# Plot PCA of constrastive activations
# ? python plot_activations.py --behaviors sycophancy --layers $(seq 0 31) --model_size "7b"

# Plot results of CAA steering effect
# ? python plot_results.py --layers $(seq 0 31) --multipliers 1 --type ab
# ? python plot_results.py --layers $(seq 0 31) --multipliers -1 0 1 --behaviors sycophancy --type ab

# Finetune a llama on a behavioral dataset using supervised finetuning on the A/B tokens
# ? python finetune_llama.py --behavior sycophancy --direction pos

# Plot similarites of steering vectors
# ? python analyze_vectors.py

