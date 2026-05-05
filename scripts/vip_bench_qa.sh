# Generate steering pairs

# python generate.py --model_name internvl3_5

# Generate steering vectors for layers of the model for a certain behavior
# python generate_vector.py --layers $(seq 0 27) --model_name internvl3_5 --behaviors refer --behavior_paths ... --output_dir ...

# Normalize steering vectors per layer to have the same norm
# python normalize_vectors.py

# Evaluate model on open-ended image referential, VIP-Bench test sets while using CAA
# python prompting_with_steering.py --behaviors refer --type vip_image_oe_qa --model_name "internvl3_5" --model_size "8b" --layers 0 --multipliers 0 
# python prompting_with_steering.py --behaviors refer --type vip_image_oe_qa --model_name "internvl3_5" --model_size "8b" --layers $(seq 0 27) --multipliers 1 2 

# Plot similarites of steering vectors
# todo python analyze_vectors.py

# Use GPT-4 to score open-ended responses
export OPEN_AI_KEY=sk-123456
export OPEN_AI_BASE="http://localhost:8000/v1"
# python vip_image_oe_qa_scoring.py