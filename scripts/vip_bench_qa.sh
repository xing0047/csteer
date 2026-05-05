# Generate steering pairs

# python generate.py \
#     --model_name "internvl2_5" \
#     --model_size "8b"

# Generate steering vectors for layers of the model for a certain behavior
# python generate_vectors.py \
#     --layers $(seq 0 31) \
#     --save_activations \
#     --model_name "internvl2_5" \
#     --model_size "8b"  \
#     --behaviors refer
# python generate_vectors.py \
#     --layers $(seq 0 47) \
#     --save_activations \
#     --model_name "internvl2_5" \
#     --model_size "26b" \
#     --behaviors refer
# python generate_vectors.py \
#     --layers $(seq 0 27) \
#     --save_activations \
#     --model_name "internvl3" \
#     --model_size "8b" \
#     --behaviors refer

# Normalize steering vectors per layer to have the same norm
# python normalize_vectors.py

# Evaluate model on open-ended image referential, VIP-Bench test sets while using CAA
# python prompting_with_steering.py --behaviors refer --type vip_image_oe_qa --model_name "internvl3" --model_size "8b" --layers 0 --multipliers 0 
# python prompting_with_steering.py --behaviors refer --type vip_image_oe_qa --model_name "internvl3" --model_size "8b" --layers $(seq 0 27) --multipliers 1 2 

# Plot PCA of constrastive activations
# todo python plot_activations.py --behaviors sycophancy --layers $(seq 0 31) --model_size "7b"

# Plot results of CAA steering effect
# todo python plot_results.py --layers $(seq 0 31) --multipliers 1 --type ab
# todo python plot_results.py --layers $(seq 0 31) --multipliers -1 0 1 --behaviors sycophancy --type ab

# Finetune a llama on a behavioral dataset using supervised finetuning on the A/B tokens
# todo python finetune_llama.py --behavior sycophancy --direction pos

# Plot similarites of steering vectors
# todo python analyze_vectors.py

# Use GPT-4 to score open-ended responses
export OPEN_AI_KEY=sk-123456
export OPEN_AI_BASE="http://localhost:8000/v1"
# python vip_image_oe_qa_scoring.py