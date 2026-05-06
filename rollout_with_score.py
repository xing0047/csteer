"""
Generate rollouts for videos/images and score them.
Generates multiple rollouts with vLLM, then uses a judge model to score them and saves results to JSON.

Supported models: internvl3, internvl3_5, qwen3vl

Supports two modes:
1. image: image mode (original)
2. video: video mode (added)

Example usage (video with internvl3_5):
python rollout_with_score.py \
    --model_name internvl3_5 \
    --model_size 8b \
    --data_type video \
    --data_path datasets/Inst-It-Dataset/inst_it_dataset_video_21k.json \
    --media_root datasets/Inst-It-Dataset \
    --n_samples 1024 \
    --num_rollouts 8 \
    --rollout_temperature 0.6 \
    --judge_base_url http://localhost:8001/v1 \
    --judge_model_name Qwen/Qwen2.5-72B-Instruct-AWQ \
    --output_dir video_rollout_exp \
    --verbose

Example usage (video with qwen3vl):
python rollout_with_score.py \
    --model_name qwen3vl \
    --model_size 8b \
    --data_type video \
    --data_path datasets/Inst-It-Dataset/inst_it_dataset_video_21k.json \
    --media_root datasets/Inst-It-Dataset \
    --n_samples 1024 \
    --num_rollouts 8 \
    --rollout_temperature 0.6 \
    --judge_base_url http://localhost:8001/v1 \
    --judge_model_name Qwen/Qwen2.5-72B-Instruct-AWQ \
    --output_dir video_rollout_exp \
    --verbose

Example usage (image):
python rollout_with_score.py \
    --model_name internvl3_5 \
    --model_size 8b \
    --data_type image \
    --data_path datasets/Inst-It-Dataset/inst_it_dataset_image_51k.json \
    --media_root datasets/Inst-It-Dataset \
    --n_samples 1024 \
    --num_rollouts 8 \
    --output_dir image_rollout_exp \
    --verbose
"""

import torch as t
from torch.utils.data import Dataset
from tqdm import tqdm
import os
import argparse
import json
import base64
from typing import List, Optional, Tuple
from openai import OpenAI
from PIL import Image
from vllm import LLM, SamplingParams
from utils.mpath_map import model_to_path_map
from utils.conversation import (
    INST_IT_IMAGE_SYSTEM_PROMPT,
    INST_IT_VIDEO_SYSTEM_PROMPT,
)
from utils.prompts import prompt_template
import re
import csv


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
        "--vllm_model_path", type=str, default=None,
        help="Path to vLLM model for generating rollouts (if None, will use model_name and model_size)"
    )
    parser.add_argument(
        "--vllm_tensor_parallel_size", type=int, default=1,
        help="Tensor parallel size for vLLM model"
    )
    parser.add_argument(
        "--vllm_gpu_memory_utilization", type=float, default=0.9,
        help="GPU memory utilization for vLLM model (default: 0.9)"
    )
    parser.add_argument(
        "--vllm_max_model_len", type=int, default=8192,
        help="Maximum model length for vLLM (default: 8192)"
    )
    parser.add_argument(
        "--judge_base_url", type=str, default="http://localhost:8001/v1",
        help="vLLM service base URL for judge model"
    )
    parser.add_argument(
        "--judge_model_name", type=str, default="Qwen/Qwen2.5-72B-Instruct-AWQ",
        help="Judge model name for evaluating rollouts"
    )
    parser.add_argument(
        "--num_rollouts", type=int, default=8,
        help="Number of rollouts to generate per sample (default: 8)"
    )
    parser.add_argument(
        "--rollout_temperature", type=float, default=0.6,
        help="Temperature for rollout generation (default: 0.6)"
    )
    parser.add_argument(
        "--output_root", type=str, default="ROLLOUT_RESULTS",
        help="Root directory for output (default: ROLLOUT_RESULTS)"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output subdirectory name for saving results"
    )
    parser.add_argument(
        '--verbose', action='store_true', help="enable verbose outputs"
    )
    return parser.parse_args()


class VideoDataset(Dataset):
    """Dataset for video samples."""
    def __init__(self, data_path, n_samples, media_root):
        with open(data_path, 'r', encoding='utf-8') as f:
            all_data = json.load(f)
        self.data = all_data[:n_samples]
        self.media_root = media_root

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Get list of frame paths
        video_path = item['video_path']  # e.g., "videos_vpt/YoutubeVIS-2021/train/05e0b0f28f"
        frame_names = [frame_info['frame_name'] for frame_info in item['frame_level_caption']]
        
        # Build absolute frame paths
        frame_paths = [os.path.join(self.media_root, video_path, frame_name) for frame_name in frame_names]
        
        caption_gt = item["video_level_caption"].strip()
        instruction = "Please describe the whole video with IDs."
        
        # Use video id as name
        video_name = os.path.basename(video_path)
        
        return frame_paths, instruction, caption_gt, video_name, idx


class ImageDataset(Dataset):
    """Dataset for image samples."""
    def __init__(self, data_path, n_samples, media_root):
        with open(data_path, 'r', encoding='utf-8') as f:
            all_data = json.load(f)
        self.data = all_data[:n_samples]
        self.media_root = media_root

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Get image path
        original_image_path = item['image_path']
        if original_image_path.startswith('images_vpt/'):
            relative_path = original_image_path[len('images_vpt/'):]
        elif original_image_path.startswith('images_raw/'):
            relative_path = original_image_path[len('images_raw/'):]
        else:
            relative_path = original_image_path
        
        image_path = os.path.join(self.media_root, "images_vpt", relative_path)
        caption_gt = item["image_level_caption"].strip()
        instruction = "Please describe the whole image with IDs."
        
        image_name = os.path.basename(relative_path)
        
        return image_path, instruction, caption_gt, image_name, idx


def generate_rollouts_for_video(
    vllm_llm: LLM,
    frame_paths: List[str],
    instruction: str,
    system_prompt: str,
    num_rollouts: int = 8,
    temperature: float = 0.6,
    max_tokens: int = 2048
) -> List[str]:
    """
    Generate multiple rollouts for a video using vLLM.
    
    Args:
        vllm_llm: vLLM LLM instance
        frame_paths: List of video frame paths
        instruction: Instruction/question
        system_prompt: System prompt
        num_rollouts: Number of rollouts to generate
        temperature: Sampling temperature
        max_tokens: Max generated tokens
    
    Returns:
        List[str]: Generated rollouts
    """
    full_prompt = f"{system_prompt}\n\n{instruction}"
    
    # Build multi-image message content
    content = []
    
    # Add each video frame as an image
    for i, frame_path in enumerate(frame_paths):
        if not os.path.isabs(frame_path):
            frame_path = os.path.abspath(frame_path)
        
        if not os.path.exists(frame_path):
            print(f"Warning: Frame not found: {frame_path}")
            continue
        
        # Use file:// URL
        frame_url = f"file://{frame_path}"
        content.append({
            "type": "image_url",
            "image_url": {"url": frame_url},
        })
    
    # Add text prompt
    content.append({
        "type": "text",
        "text": full_prompt,
    })
    
    messages = [{"role": "user", "content": content}]
    
    sampling_params = SamplingParams(
        n=num_rollouts,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    
    try:
        request_outputs = vllm_llm.chat(messages, sampling_params=sampling_params)
        
        rollouts = []
        if isinstance(request_outputs, list):
            for request_output in request_outputs:
                if hasattr(request_output, 'outputs') and request_output.outputs:
                    for output in request_output.outputs:
                        if hasattr(output, 'text'):
                            rollouts.append(output.text.strip())
                elif hasattr(request_output, 'text'):
                    rollouts.append(request_output.text.strip())
        elif hasattr(request_outputs, 'outputs') and request_outputs.outputs:
            for output in request_outputs.outputs:
                if hasattr(output, 'text'):
                    rollouts.append(output.text.strip())
        elif hasattr(request_outputs, 'text'):
            rollouts.append(request_outputs.text.strip())
        else:
            print(f"Warning: Unexpected output format from vLLM")
            rollouts = [""] * num_rollouts
        
        while len(rollouts) < num_rollouts:
            rollouts.append("")
        
        return rollouts[:num_rollouts]
        
    except Exception as e:
        print(f"Error generating rollouts: {e}")
        import traceback
        traceback.print_exc()
        return [""] * num_rollouts


def generate_rollouts_for_image(
    vllm_llm: LLM,
    image_path: str,
    instruction: str,
    system_prompt: str,
    num_rollouts: int = 8,
    temperature: float = 0.6,
    max_tokens: int = 1024
) -> List[str]:
    """
    Generate multiple rollouts for an image using vLLM.
    """
    full_prompt = f"{system_prompt}\n\n{instruction}"
    
    if not os.path.isabs(image_path):
        image_path = os.path.abspath(image_path)
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    image_url = f"file://{image_path}"
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": full_prompt},
            ],
        }
    ]
    
    sampling_params = SamplingParams(
        n=num_rollouts,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    
    try:
        request_outputs = vllm_llm.chat(messages, sampling_params=sampling_params)
        
        rollouts = []
        if isinstance(request_outputs, list):
            for request_output in request_outputs:
                if hasattr(request_output, 'outputs') and request_output.outputs:
                    for output in request_output.outputs:
                        if hasattr(output, 'text'):
                            rollouts.append(output.text.strip())
                elif hasattr(request_output, 'text'):
                    rollouts.append(request_output.text.strip())
        elif hasattr(request_outputs, 'outputs') and request_outputs.outputs:
            for output in request_outputs.outputs:
                if hasattr(output, 'text'):
                    rollouts.append(output.text.strip())
        
        while len(rollouts) < num_rollouts:
            rollouts.append("")
        
        return rollouts[:num_rollouts]
        
    except Exception as e:
        print(f"Error generating rollouts: {e}")
        import traceback
        traceback.print_exc()
        return [""] * num_rollouts


def extract_score(text: str) -> float:
    """Extract a score (0-1) from judge model output."""
    for pat in [r"\b([01](?:\.\d+)?)\b", r"\b(0?\.\d+)\b"]:
        m = re.findall(pat, text)
        if m:
            try:
                v = float(m[0])
                if 0.0 <= v <= 1.0:
                    return v
            except Exception:
                pass
    try:
        v = float(text.strip())
        if 0.0 <= v <= 1.0:
            return v
    except Exception:
        pass
    print(f"[Warn] cannot extract score from: {text!r}")
    return 0.0


def judge_rollout_with_model(
    judge_client: OpenAI,
    judge_model_name: str,
    instruction: str,
    rollout_text: str,
    ground_truth: str,
    data_type: str = "video"
) -> float:
    """
    Score a rollout using the judge model.
    """
    # Select prompt template based on data type
    if "gpt_eval_rollout" in prompt_template and data_type in prompt_template["gpt_eval_rollout"]:
        eval_prompt_template = prompt_template["gpt_eval_rollout"][data_type]
    elif "gpt_eval" in prompt_template and data_type in prompt_template["gpt_eval"]:
        eval_prompt_template = prompt_template["gpt_eval"][data_type]
    else:
        # Fall back to the image template
        eval_prompt_template = prompt_template.get("gpt_eval", {}).get("image", "")
    
    eval_input = f"question: {instruction}\nground truth answer for the question: {ground_truth}\nresponse from the tester: {rollout_text}"
    full_prompt = f"{eval_prompt_template}\n{eval_input}"
    
    messages = [{"role": "user", "content": full_prompt}]
    
    try:
        response = judge_client.chat.completions.create(
            model=judge_model_name,
            messages=messages,
            temperature=0.0,
            max_tokens=128,
        )
        judgment_text = response.choices[0].message.content.strip()
        score = extract_score(judgment_text)
        return score
    except Exception as e:
        print(f"Error judging rollout: {e}")
        return 0.0


def main():
    args = parse_args()
    
    # Output directory
    output_dir = os.path.join(args.output_root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Resolve model path
    if args.vllm_model_path is None:
        vllm_model_path = model_to_path_map[args.model_name][args.model_size]
    else:
        vllm_model_path = args.vllm_model_path
    
    # Absolute media root path
    media_root_abs = os.path.abspath(args.media_root)
    
    print("=" * 50)
    print("Generating rollouts and scoring")
    print("=" * 50)
    print(f"Data type: {args.data_type}")
    print(f"Model: {args.model_name}_{args.model_size}")
    print(f"Data path: {args.data_path}")
    print(f"Media root: {media_root_abs}")
    print(f"Num samples: {args.n_samples}")
    print(f"Num rollouts: {args.num_rollouts}")
    print(f"Output dir: {output_dir}")
    print("=" * 50)
    
    # Load vLLM model
    print(f"\nLoading vLLM model: {vllm_model_path}")
    vllm_llm = LLM(
        model=vllm_model_path,
        tensor_parallel_size=args.vllm_tensor_parallel_size,
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        max_model_len=args.vllm_max_model_len,
        enforce_eager=True,
        trust_remote_code=True,
        allowed_local_media_path="/",
        limit_mm_per_prompt={"image": 32} if args.data_type == "video" else {"image": 1},
    )
    print("vLLM model loaded")
    
    # Connect to judge service
    judge_client = OpenAI(base_url=args.judge_base_url, api_key="EMPTY", timeout=600)
    print(f"Connected to judge service: {args.judge_base_url}")
    
    # Create dataset
    if args.data_type == "video":
        dataset = VideoDataset(args.data_path, args.n_samples, media_root_abs)
        system_prompt = INST_IT_VIDEO_SYSTEM_PROMPT
    else:
        dataset = ImageDataset(args.data_path, args.n_samples, media_root_abs)
        system_prompt = INST_IT_IMAGE_SYSTEM_PROMPT
    
    print(f"Dataset size: {len(dataset)}")
    
    # Store results
    all_results = []
    
    skipped_samples = 0
    
    for idx in tqdm(range(len(dataset)), desc="Processing samples"):
        if args.data_type == "video":
            frame_paths, instruction, caption_gt, media_name, sample_idx = dataset[idx]
            
            # Check frames exist
            valid_frames = [p for p in frame_paths if os.path.exists(p)]
            if len(valid_frames) == 0:
                if args.verbose:
                    print(f"Warning: No valid frames for video {media_name}, skipping...")
                skipped_samples += 1
                continue
            
            # Generate rollouts
            rollouts = generate_rollouts_for_video(
                vllm_llm=vllm_llm,
                frame_paths=valid_frames,
                instruction=instruction,
                system_prompt=system_prompt,
                num_rollouts=args.num_rollouts,
                temperature=args.rollout_temperature,
            )
        else:
            image_path, instruction, caption_gt, media_name, sample_idx = dataset[idx]
            
            if not os.path.exists(image_path):
                if args.verbose:
                    print(f"Warning: Image not found: {image_path}, skipping...")
                skipped_samples += 1
                continue
            
            rollouts = generate_rollouts_for_image(
                vllm_llm=vllm_llm,
                image_path=image_path,
                instruction=instruction,
                system_prompt=system_prompt,
                num_rollouts=args.num_rollouts,
                temperature=args.rollout_temperature,
            )
        
        # Score
        rollout_details = []
        for rollout_idx, rollout in enumerate(rollouts):
            if not rollout:
                score = 0.0
            else:
                score = judge_rollout_with_model(
                    judge_client=judge_client,
                    judge_model_name=args.judge_model_name,
                    instruction=instruction,
                    rollout_text=rollout,
                    ground_truth=caption_gt,
                    data_type=args.data_type
                )
            
            rollout_details.append({
                'rollout_id': rollout_idx,
                'rollout_text': rollout,
                'score': score
            })
        
        num_incorrect = sum(1 for r in rollout_details if r['score'] <= 0.6)
        
        result = {
            'sample_id': sample_idx,
            'media_name': media_name,
            'ground_truth': caption_gt,
            'num_incorrect': num_incorrect,
            'rollouts': rollout_details
        }
        
        if args.data_type == "video":
            result['frame_paths'] = [os.path.relpath(p, media_root_abs) for p in valid_frames]
        else:
            result['image_path'] = os.path.relpath(image_path, media_root_abs) if os.path.exists(image_path) else None
        
        all_results.append(result)
        
        if args.verbose:
            scores = [r['score'] for r in rollout_details]
            print(f"\n[{idx+1}/{len(dataset)}] {media_name}")
            print(f"  Scores: {[f'{s:.2f}' for s in scores]}")
            print(f"  Incorrect: {num_incorrect}/{args.num_rollouts}")
    
    # Save JSON results
    json_path = os.path.join(output_dir, "judge_results.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\nJSON results saved to: {json_path}")
    
    # Save CSV results
    csv_path = os.path.join(output_dir, "judge_results.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['sample_idx', 'media_name'] + [f'rollout_{i}' for i in range(args.num_rollouts)] + ['num_incorrect']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in sorted(all_results, key=lambda x: x['sample_id']):
            row = {
                'sample_idx': result['sample_id'],
                'media_name': result['media_name']
            }
            for rollout_info in result['rollouts']:
                rollout_id = rollout_info['rollout_id']
                score = rollout_info['score']
                row[f'rollout_{rollout_id}'] = f'{score:.1f}'
            row['num_incorrect'] = result['num_incorrect']
            writer.writerow(row)
    
    print(f"CSV results saved to: {csv_path}")
    print(f"Total samples: {len(all_results)}")
    
    # Summary stats
    total_rollouts = len(all_results) * args.num_rollouts
    total_incorrect = sum(r['num_incorrect'] for r in all_results)
    print(f"Total rollouts: {total_rollouts}")
    if total_rollouts > 0:
        print(f"Incorrect rollouts: {total_incorrect} ({100*total_incorrect/total_rollouts:.1f}%)")
    
if __name__ == "__main__":
    main()
