"""
Generate steering vectors for GT vs. Rollout (supports video and image).
The vector is computed from activation differences between the official ground truth (positive) and the original incorrect rollout (negative).

Supported models: internvl3, internvl3_5, qwen3vl

Supports two modes:
1. image: image mode
2. video: video mode

Example usage (video with internvl3_5):
python generate_vector_gt.py \
    --model_name internvl3_5 \
    --model_size 8b \
    --data_type video \
    --layers $(seq 0 35) \
    --judge_results_json ./ROLLOUT_RESULTS/video_rollout_exp/judge_results.json \
    --data_path ../DATA/Inst-It-Dataset/inst_it_dataset_video_21k.json \
    --media_root ../DATA/Inst-It-Dataset \
    --n_samples 1024 \
    --score_threshold 0.6 \
    --output_dir refer_gt_video_exp \
    --use_flash_attn \
    --verbose

Example usage (video with qwen3vl):
python generate_vector_gt.py \
    --model_name qwen3vl \
    --model_size 8b \
    --data_type video \
    --layers $(seq 0 35) \
    --judge_results_json ./ROLLOUT_RESULTS/video_rollout_exp/judge_results.json \
    --data_path ../DATA/Inst-It-Dataset/inst_it_dataset_video_21k.json \
    --media_root ../DATA/Inst-It-Dataset \
    --n_samples 1024 \
    --score_threshold 0.6 \
    --output_dir refer_gt_video_exp \
    --use_flash_attn \
    --verbose

Example usage (image):
python generate_vector_gt.py \
    --model_name internvl3_5 \
    --model_size 8b \
    --data_type image \
    --layers $(seq 0 35) \
    --judge_results_json ./ROLLOUT_RESULTS/image_rollout_exp/judge_results.json \
    --data_path ../DATA/Inst-It-Dataset/inst_it_dataset_image_51k.json \
    --media_root ../DATA/Inst-It-Dataset \
    --n_samples 1024 \
    --output_dir refer_gt_image_exp \
    --use_flash_attn \
    --verbose
"""

import torch as t
from torch.utils.data import Dataset
from tqdm import tqdm
import os
import argparse
import json
from typing import List, Optional, Tuple
from PIL import Image
from behaviors import (
    get_vector_dir,
    get_vector_path,
    REFER_VPT,
)
from utils.mwrapper_map import (
    model_to_wrapper_map
)
from utils.mpath_map import model_to_path_map
from utils.conversation import (
    INST_IT_IMAGE_SYSTEM_PROMPT,
    INST_IT_VIDEO_SYSTEM_PROMPT,
)
import gc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, choices=["internvl3", "internvl3_5", "qwen3vl"], default="internvl3_5"
    )
    parser.add_argument(
        "--model_size", type=str, choices=["2b", "8b", "32b", "38b"]
    )
    parser.add_argument(
        "--data_type", type=str, choices=["image", "video"], default="video",
        help="Data type: image or video"
    )
    parser.add_argument(
        "--layers", nargs="+", type=int, default=list(range(36))
    )
    parser.add_argument(
        '--use_flash_attn', action='store_true', help="enable flash attention"
    )
    parser.add_argument(
        '--verbose', action='store_true', help="enable verbose outputs"
    )
    parser.add_argument(
        "--judge_results_json", type=str, required=True,
        help="Path to judge_results.json (contains original rollouts with scores and ground truth)"
    )
    parser.add_argument(
        "--data_path", type=str, required=True,
        help="Path to dataset JSON file"
    )
    parser.add_argument(
        "--media_root", type=str, required=True,
        help="Root directory containing images_vpt or videos_vpt folder"
    )
    parser.add_argument(
        "--n_samples", type=int, default=1024,
        help="Number of samples to use (default: 1024)"
    )
    parser.add_argument(
        "--score_threshold", type=float, default=0.6,
        help="Score threshold for selecting rollouts (default: 0.6)"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory name for saving vectors"
    )
    return parser.parse_args()


class GTRolloutDataset(Dataset):
    """Dataset for loading matched GT-rollout pairs (supports image and video)"""
    
    def __init__(self, judge_results, data_path, media_root, 
                 data_type="video", score_threshold=0.6, n_samples=1024):
        """
        Args:
            judge_results: List of dicts from judge_results.json
            data_path: Path to dataset JSON file
            media_root: Root directory containing images_vpt or videos_vpt folder
            data_type: "image" or "video"
            score_threshold: Only include rollouts with score <= threshold
            n_samples: Maximum number of GT samples to include (all rollouts from these GTs will be included)
        """
        with open(data_path, 'r', encoding='utf-8') as f:
            self.raw_data = json.load(f)
        
        self.media_root = os.path.abspath(media_root)
        self.data_type = data_type
        
        # Build indices
        judge_dict = {item['sample_id']: item for item in judge_results}
        
        # Select first n_samples GTs (sample_id)
        sorted_sample_ids = sorted(judge_dict.keys())
        if n_samples > 0:
            selected_sample_ids = sorted_sample_ids[:n_samples]
        else:
            selected_sample_ids = sorted_sample_ids
        
        # Match and filter: only keep the first n_samples GTs, but include all qualifying rollouts for those GTs.
        self.samples = []
        for sample_id in selected_sample_ids:
            judge_item = judge_dict[sample_id]
            
            media_name = judge_item.get('media_name') or judge_item.get('image_name', '')
            ground_truth = judge_item['ground_truth']
            
            # Resolve media path based on data type
            if data_type == "video":
                # Use frame paths from judge_results if available; otherwise fall back to raw dataset.
                if 'frame_paths' in judge_item:
                    frame_paths = [os.path.join(self.media_root, p) for p in judge_item['frame_paths']]
                elif sample_id < len(self.raw_data):
                    raw_item = self.raw_data[sample_id]
                    video_path = raw_item['video_path']
                    frame_names = [frame_info['frame_name'] for frame_info in raw_item['frame_level_caption']]
                    frame_paths = [os.path.join(self.media_root, video_path, fn) for fn in frame_names]
                else:
                    continue
                media_path = frame_paths
                instruction = "Please describe the whole video with IDs."
            else:
                # Image mode
                if 'image_path' in judge_item and judge_item['image_path']:
                    image_path = os.path.join(self.media_root, judge_item['image_path'])
                elif sample_id < len(self.raw_data):
                    raw_item = self.raw_data[sample_id]
                    original_image_path = raw_item.get('image_path', '')
                    if original_image_path.startswith('images_vpt/'):
                        relative_path = original_image_path[len('images_vpt/'):]
                    elif original_image_path.startswith('images_raw/'):
                        relative_path = original_image_path[len('images_raw/'):]
                    else:
                        relative_path = original_image_path
                    image_path = os.path.join(self.media_root, "images_vpt", relative_path)
                else:
                    continue
                media_path = image_path
                instruction = "Please describe the whole image with IDs."
            
            # Match rollout pairs
            for rollout in judge_item['rollouts']:
                rollout_id = rollout['rollout_id']
                original_score = rollout['score']
                original_text = rollout['rollout_text']
                
                if original_score > score_threshold:
                    continue
                
                # Use GT and rollout directly; rewritten_rollouts are not needed here.
                self.samples.append({
                    'sample_id': sample_id,
                    'rollout_id': rollout_id,
                    'media_name': media_name,
                    'media_path': media_path,
                    'instruction': instruction,
                    'ground_truth': ground_truth,
                    'original_rollout': original_text,
                    'original_score': original_score
                })
        
        # Do not truncate pair list: GT count is already limited, and all qualifying rollouts under those GTs are included.
        
        print(f"Loaded {len(self.samples)} matched GT-rollout pairs")
        print(f"Data type: {data_type}")
        print(f"Score threshold: {score_threshold}")
        print(f"Number of GT samples: {len(selected_sample_ids)}")
        print(f"Total rollout pairs: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def generate_save_vectors(
    layers: List[int],
    wrapper,
    model_name: str,
    model_size: str,
    use_flash_attn: bool,
    dataset: GTRolloutDataset,
    data_type: str,
    output_dir: str,
    verbose: bool = False,
):
    """
    Generate and save steering vectors for GT vs Rollout.
    The vector is computed as (GT activations - rollout activations).
    """
    behavior = REFER_VPT
    vector_dir = get_vector_dir(behavior, model_name, model_size, output_dir)
    if not os.path.exists(vector_dir):
        os.makedirs(vector_dir)
    
    # Load model wrapper
    print(f"Loading wrapper model...")
    model = wrapper(
        name=model_name,
        size=model_size,
        use_flash_attn=use_flash_attn
    )
    
    model.set_save_internal_decodings(False)
    model.reset_all()
    
    # Note: get_tokens_for_compare will automatically add the system prompt
    # (INST_IT_IMAGE_SYSTEM_PROMPT or INST_IT_VIDEO_SYSTEM_PROMPT).
    
    # Store activations
    gt_activations = dict([(layer, []) for layer in layers])
    original_rollout_activations = dict([(layer, []) for layer in layers])
    
    for idx, sample in enumerate(tqdm(dataset, desc="Processing samples")):
        media_path = sample['media_path']
        instruction = sample['instruction']
        original_rollout = sample['original_rollout']
        ground_truth = sample['ground_truth']
        
        # Check media exists
        if data_type == "video":
            valid_frames = [p for p in media_path if os.path.exists(p)]
            if len(valid_frames) == 0:
                if verbose:
                    print(f"Warning: No valid frames for {sample['media_name']}, skipping...")
                continue
            media_for_model = valid_frames
            media_type = "video"
        else:
            if not os.path.exists(media_path):
                if verbose:
                    print(f"Warning: Image not found: {media_path}, skipping...")
                continue
            media_for_model = media_path
            media_type = "image"
        
        if verbose and idx % 100 == 0:
            print(f"\n[{idx+1}/{len(dataset)}] Processing: {sample['media_name']}")
            print(f"  Original score: {sample['original_score']:.2f}")
        
        # 1. Extract positive activations (ground truth)
        model.reset_all()
        try:
            tokens_gt = model.get_tokens_for_compare(
                media_for_model, media_type, instruction, ground_truth,
                verbose=False
            )
            if isinstance(tokens_gt, t.Tensor):
                tokens_gt = tokens_gt.unsqueeze(0).to(model.device)
            else:
                tokens_gt = t.tensor(tokens_gt, dtype=t.long).unsqueeze(0).to(model.device)
            
            model.get_logits(media_for_model, media_type, tokens_gt)
            for layer in layers:
                gt_act = model.get_last_activations(layer)
                if len(gt_act.shape) == 3:
                    gt_act = gt_act.squeeze(0)
                assert len(gt_act.shape) == 2, f"Expected 2D activations, got shape {gt_act.shape}"
                gt_act = gt_act[-2, :].detach().cpu()
                gt_activations[layer].append(gt_act)
        except Exception as e:
            if verbose:
                print(f"Error processing ground truth: {e}")
            continue
        
        del tokens_gt
        model.reset_all()
        t.cuda.empty_cache()
        
        # 2. Extract negative activations (original rollout)
        model.reset_all()
        try:
            tokens_original = model.get_tokens_for_compare(
                media_for_model, media_type, instruction, original_rollout,
                verbose=False
            )
            if isinstance(tokens_original, t.Tensor):
                tokens_original = tokens_original.unsqueeze(0).to(model.device)
            else:
                tokens_original = t.tensor(tokens_original, dtype=t.long).unsqueeze(0).to(model.device)
            
            model.get_logits(media_for_model, media_type, tokens_original)
            for layer in layers:
                original_act = model.get_last_activations(layer)
                if len(original_act.shape) == 3:
                    original_act = original_act.squeeze(0)
                assert len(original_act.shape) == 2, f"Expected 2D activations, got shape {original_act.shape}"
                original_act = original_act[-2, :].detach().cpu()
                original_rollout_activations[layer].append(original_act)
        except Exception as e:
            if verbose:
                print(f"Error processing original rollout: {e}")
            # Remove corresponding GT activations to keep list lengths aligned.
            for layer in layers:
                if len(gt_activations[layer]) > len(original_rollout_activations[layer]):
                    gt_activations[layer].pop()
            continue
        
        del tokens_original
        model.reset_all()
        t.cuda.empty_cache()
        
        if (idx + 1) % 50 == 0:
            gc.collect()
            t.cuda.empty_cache()
    
    # 3. Compute vectors: GT - Rollout
    print("\nComputing steering vectors...")
    num_pairs = len(gt_activations[layers[0]])
    print(f"Total pairs: {num_pairs}")
    
    if num_pairs == 0:
        print("Error: No valid pairs found!")
        return
    
    for layer in tqdm(layers, desc="Computing vectors"):
        all_gt = t.stack(gt_activations[layer], dim=0)
        all_original = t.stack(original_rollout_activations[layer], dim=0)
        
        # vec = mean(GT - Rollout)
        vec = (all_gt - all_original).mean(dim=0)
        
        vec_path = get_vector_path(behavior, model_name, model_size, output_dir, layer)
        t.save(vec, vec_path)
        
        if verbose:
            print(f"  Layer {layer}: vector shape {vec.shape}")
    
    print(f"\nAll vectors saved to: {vector_dir}")
    print(f"Total pairs used: {num_pairs}")


def main():
    args = parse_args()
    
    # Load data
    print("Loading judge results...")
    with open(args.judge_results_json, 'r', encoding='utf-8') as f:
        judge_results = json.load(f)
    
    print(f"Loaded {len(judge_results)} judge results")
    
    # Create dataset
    dataset = GTRolloutDataset(
        judge_results=judge_results,
        data_path=args.data_path,
        media_root=args.media_root,
        data_type=args.data_type,
        score_threshold=args.score_threshold,
        n_samples=args.n_samples
    )
    
    if len(dataset) == 0:
        print("Error: No matched samples found!")
        return
    
    # Get model wrapper
    wrapper = model_to_wrapper_map[args.model_name][args.model_size]
    
    # Generate and save vectors
    generate_save_vectors(
        layers=args.layers,
        wrapper=wrapper,
        model_name=args.model_name,
        model_size=args.model_size,
        use_flash_attn=args.use_flash_attn,
        dataset=dataset,
        data_type=args.data_type,
        output_dir=args.output_dir,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
