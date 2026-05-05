"""
为 Rewrite v.s. Rollout 生成 steering vectors（支持视频和图像）。
通过对比改写后的rollout（正样本）和原始错误的rollout（负样本）的激活差异来生成向量。

支持的模型：internvl3, internvl3_5, qwen3vl

支持两种模式：
1. image: 图像模式
2. video: 视频模式

Example usage (video with internvl3_5):
python generate_vector_rewrite.py \
    --model_name internvl3_5 \
    --model_size 8b \
    --data_type video \
    --layers $(seq 0 35) \
    --judge_results_json ./ROLLOUT_RESULTS/video_rollout_exp/judge_results.json \
    --rewritten_rollouts_json ./ROLLOUT_RESULTS/video_rollout_exp/rewritten_rollouts.json \
    --data_path ../DATA/Inst-It-Dataset/inst_it_dataset_video_21k.json \
    --media_root ../DATA/Inst-It-Dataset \
    --n_samples 1024 \
    --score_threshold 0.6 \
    --output_dir refer_rewrite_video_exp \
    --use_flash_attn \
    --verbose

Example usage (video with qwen3vl):
python generate_vector_rewrite.py \
    --model_name qwen3vl \
    --model_size 8b \
    --data_type video \
    --layers $(seq 0 35) \
    --judge_results_json ./ROLLOUT_RESULTS/video_rollout_exp/judge_results.json \
    --rewritten_rollouts_json ./ROLLOUT_RESULTS/video_rollout_exp/rewritten_rollouts.json \
    --data_path ../DATA/Inst-It-Dataset/inst_it_dataset_video_21k.json \
    --media_root ../DATA/Inst-It-Dataset \
    --n_samples 1024 \
    --score_threshold 0.6 \
    --output_dir refer_rewrite_video_exp \
    --use_flash_attn \
    --verbose

Example usage (image):
python generate_vector_rewrite.py \
    --model_name internvl3_5 \
    --model_size 8b \
    --data_type image \
    --layers $(seq 0 35) \
    --judge_results_json ./ROLLOUT_RESULTS/image_rollout_exp/judge_results.json \
    --rewritten_rollouts_json ./ROLLOUT_RESULTS/image_rollout_exp/rewritten_rollouts.json \
    --data_path ../DATA/Inst-It-Dataset/inst_it_dataset_image_51k.json \
    --media_root ../DATA/Inst-It-Dataset \
    --n_samples 1024 \
    --output_dir refer_rewrite_image_exp \
    --use_flash_attn \
    --verbose
"""

import torch as t
from torch.utils.data import Dataset
from tqdm import tqdm
import os
import argparse
import json
import time
import psutil
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
        help="Path to judge_results.json (contains original rollouts with scores)"
    )
    parser.add_argument(
        "--rewritten_rollouts_json", type=str, required=True,
        help="Path to rewritten_rollouts.json (contains rewritten rollouts)"
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
    parser.add_argument(
        '--track_cost', action='store_true',
        help="Track and save cost metrics (time, forward passes, GPU memory, etc.) to a JSON file"
    )
    return parser.parse_args()


class RewriteRolloutDataset(Dataset):
    """Dataset for loading matched rewrite-rollout pairs (supports image and video)"""
    
    def __init__(self, judge_results, rewritten_rollouts, data_path, media_root, 
                 data_type="video", score_threshold=0.6, n_samples=1024):
        """
        Args:
            judge_results: List of dicts from judge_results.json
            rewritten_rollouts: List of dicts from rewritten_rollouts.json
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
        
        # 创建索引
        judge_dict = {item['sample_id']: item for item in judge_results}
        rewritten_dict = {item['sample_id']: item for item in rewritten_rollouts}
        
        # 获取前 n_samples 个 GT（sample_id），这些GT必须在rewritten_dict中存在
        sorted_sample_ids = sorted(judge_dict.keys())
        # 先筛选出在rewritten_dict中存在的sample_id
        available_sample_ids = [sid for sid in sorted_sample_ids if sid in rewritten_dict]
        if n_samples > 0:
            selected_sample_ids = available_sample_ids[:n_samples]
        else:
            selected_sample_ids = available_sample_ids
        
        # 匹配并筛选数据：只处理前 n_samples 个 GT，但包含它们的所有符合条件的 rollout
        self.samples = []
        for sample_id in selected_sample_ids:
            
            judge_item = judge_dict[sample_id]
            rewritten_item = rewritten_dict[sample_id]
            
            media_name = judge_item.get('media_name') or judge_item.get('image_name', '')
            ground_truth = judge_item['ground_truth']
            
            # 根据数据类型获取媒体路径
            if data_type == "video":
                # 从judge_results获取帧路径，或从原始数据获取
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
                # 图像模式
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
            
            # 匹配rollout pairs
            for rollout in judge_item['rollouts']:
                rollout_id = rollout['rollout_id']
                original_score = rollout['score']
                original_text = rollout['rollout_text']
                
                if original_score > score_threshold:
                    continue
                
                # 查找对应的rewritten rollout
                rewritten_rollout = None
                for r_rollout in rewritten_item['rollouts']:
                    if r_rollout['rollout_id'] == rollout_id:
                        rewritten_rollout = r_rollout['rollout_text']
                        break
                
                if rewritten_rollout:
                    self.samples.append({
                        'sample_id': sample_id,
                        'rollout_id': rollout_id,
                        'media_name': media_name,
                        'media_path': media_path,
                        'instruction': instruction,
                        'ground_truth': ground_truth,
                        'original_rollout': original_text,
                        'rewritten_rollout': rewritten_rollout,
                        'original_score': original_score
                    })
        
        # 不再截断样本对列表，因为已经限制了GT的数量
        # 前 n_samples 个 GT 下的所有符合条件的 rollout 都会被包含
        
        print(f"Loaded {len(self.samples)} matched rewrite-rollout pairs")
        print(f"Data type: {data_type}")
        print(f"Score threshold: {score_threshold}")
        print(f"Number of GT samples: {len(selected_sample_ids)}")
        print(f"Total rollout pairs: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def _get_gpu_memory_info():
    """获取当前GPU显存使用信息 (MB)"""
    info = {}
    if t.cuda.is_available():
        for i in range(t.cuda.device_count()):
            allocated = t.cuda.memory_allocated(i) / 1024**2
            reserved = t.cuda.memory_reserved(i) / 1024**2
            props = t.cuda.get_device_properties(i)
            total_bytes = getattr(props, "total_memory", None)
            if total_bytes is None:
                total_bytes = getattr(props, "total_mem", 0)
            total = total_bytes / 1024**2
            info[f"gpu_{i}"] = {
                "allocated_MB": round(allocated, 2),
                "reserved_MB": round(reserved, 2),
                "total_MB": round(total, 2),
            }
    return info


def generate_save_vectors(
    layers: List[int],
    wrapper,
    model_name: str,
    model_size: str,
    use_flash_attn: bool,
    dataset: RewriteRolloutDataset,
    data_type: str,
    output_dir: str,
    verbose: bool = False,
    track_cost: bool = False,
):
    """
    生成并保存 Rewrite vs Rollout 的 steering vectors
    """
    cost_data = {}
    t_total_start = time.time()
    
    behavior = REFER_VPT
    vector_dir = get_vector_dir(behavior, model_name, model_size, output_dir)
    if not os.path.exists(vector_dir):
        os.makedirs(vector_dir)
    
    # 加载模型wrapper
    print(f"Loading wrapper model...")
    t_model_load_start = time.time()
    model = wrapper(
        name=model_name,
        size=model_size,
        use_flash_attn=use_flash_attn
    )
    t_model_load_end = time.time()
    
    model.set_save_internal_decodings(False)
    model.reset_all()
    
    if track_cost:
        cost_data["model_load_time_s"] = round(t_model_load_end - t_model_load_start, 3)
        cost_data["gpu_after_model_load"] = _get_gpu_memory_info()
    
    # 存储激活
    rewritten_rollout_activations = dict([(layer, []) for layer in layers])
    original_rollout_activations = dict([(layer, []) for layer in layers])
    
    forward_pass_times = []
    skipped_samples = 0
    error_samples = 0
    
    t_extract_start = time.time()
    for idx, sample in enumerate(tqdm(dataset, desc="Processing samples")):
        media_path = sample['media_path']
        instruction = sample['instruction']
        original_rollout = sample['original_rollout']
        rewritten_rollout = sample['rewritten_rollout']
        
        # 检查媒体文件
        if data_type == "video":
            valid_frames = [p for p in media_path if os.path.exists(p)]
            if len(valid_frames) == 0:
                if verbose:
                    print(f"Warning: No valid frames for {sample['media_name']}, skipping...")
                skipped_samples += 1
                continue
            media_for_model = valid_frames
            media_type = "video"
        else:
            if not os.path.exists(media_path):
                if verbose:
                    print(f"Warning: Image not found: {media_path}, skipping...")
                skipped_samples += 1
                continue
            media_for_model = media_path
            media_type = "image"
        
        if verbose and idx % 100 == 0:
            print(f"\n[{idx+1}/{len(dataset)}] Processing: {sample['media_name']}")
            print(f"  Original score: {sample['original_score']:.2f}")
        
        t_fwd_start = time.time()
        
        # 1. 提取正样本激活（rewritten rollout）
        model.reset_all()
        try:
            tokens_rewritten = model.get_tokens_for_compare(
                media_for_model, media_type, instruction, rewritten_rollout,
                verbose=False
            )
            if isinstance(tokens_rewritten, t.Tensor):
                tokens_rewritten = tokens_rewritten.unsqueeze(0).to(model.device)
            else:
                tokens_rewritten = t.tensor(tokens_rewritten, dtype=t.long).unsqueeze(0).to(model.device)
            
            model.get_logits(media_for_model, media_type, tokens_rewritten)
            for layer in layers:
                rewritten_act = model.get_last_activations(layer)
                if len(rewritten_act.shape) == 3:
                    rewritten_act = rewritten_act.squeeze(0)
                assert len(rewritten_act.shape) == 2, f"Expected 2D activations, got shape {rewritten_act.shape}"
                rewritten_act = rewritten_act[-2, :].detach().cpu()
                rewritten_rollout_activations[layer].append(rewritten_act)
        except Exception as e:
            if verbose:
                print(f"Error processing rewritten rollout: {e}")
            error_samples += 1
            continue
        
        del tokens_rewritten
        model.reset_all()
        t.cuda.empty_cache()
        
        # 2. 提取负样本激活（original rollout）
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
            for layer in layers:
                if len(rewritten_rollout_activations[layer]) > len(original_rollout_activations[layer]):
                    rewritten_rollout_activations[layer].pop()
            error_samples += 1
            continue
        
        t_fwd_end = time.time()
        if track_cost:
            forward_pass_times.append(t_fwd_end - t_fwd_start)
        
        del tokens_original
        model.reset_all()
        t.cuda.empty_cache()
        
        if (idx + 1) % 50 == 0:
            gc.collect()
            t.cuda.empty_cache()
    
    t_extract_end = time.time()
    
    # 3. 计算向量
    print("\nComputing steering vectors...")
    num_pairs = len(rewritten_rollout_activations[layers[0]])
    print(f"Total pairs: {num_pairs}")
    
    if num_pairs == 0:
        print("Error: No valid pairs found!")
        return
    
    t_compute_start = time.time()
    for layer in tqdm(layers, desc="Computing vectors"):
        all_rewritten = t.stack(rewritten_rollout_activations[layer], dim=0)
        all_original = t.stack(original_rollout_activations[layer], dim=0)
        
        vec = (all_rewritten - all_original).mean(dim=0)
        
        vec_path = get_vector_path(behavior, model_name, model_size, output_dir, layer)
        t.save(vec, vec_path)
        
        if verbose:
            print(f"  Layer {layer}: vector shape {vec.shape}")
    t_compute_end = time.time()
    
    print(f"\nAll vectors saved to: {vector_dir}")
    print(f"Total pairs used: {num_pairs}")
    
    t_total_end = time.time()
    
    # 保存开销统计
    if track_cost:
        cost_data["step"] = "generate_vector_rewrite"
        cost_data["model"] = f"{model_name}_{model_size}"
        cost_data["data_type"] = data_type
        cost_data["num_layers"] = len(layers)
        cost_data["layers"] = layers
        cost_data["total_dataset_size"] = len(dataset)
        cost_data["valid_pairs_used"] = num_pairs
        cost_data["skipped_samples"] = skipped_samples
        cost_data["error_samples"] = error_samples
        cost_data["total_forward_passes"] = num_pairs * 2
        cost_data["total_wall_time_s"] = round(t_total_end - t_total_start, 3)
        cost_data["activation_extraction"] = {
            "total_time_s": round(t_extract_end - t_extract_start, 3),
            "avg_time_per_pair_s": round(sum(forward_pass_times) / len(forward_pass_times), 3) if forward_pass_times else 0,
            "min_time_per_pair_s": round(min(forward_pass_times), 3) if forward_pass_times else 0,
            "max_time_per_pair_s": round(max(forward_pass_times), 3) if forward_pass_times else 0,
        }
        cost_data["vector_computation"] = {
            "time_s": round(t_compute_end - t_compute_start, 3),
        }
        cost_data["gpu_after_inference"] = _get_gpu_memory_info()
        cost_data["system"] = {
            "cpu_count": psutil.cpu_count(),
            "ram_total_GB": round(psutil.virtual_memory().total / 1024**3, 2),
            "ram_used_GB": round(psutil.virtual_memory().used / 1024**3, 2),
        }
        
        cost_json_path = os.path.join(vector_dir, "cost_generate_vector.json")
        with open(cost_json_path, 'w', encoding='utf-8') as f:
            json.dump(cost_data, f, ensure_ascii=False, indent=2)
        print(f"\n开销统计已保存到: {cost_json_path}")


def main():
    args = parse_args()
    
    # 加载数据
    print("Loading judge results and rewritten rollouts...")
    with open(args.judge_results_json, 'r', encoding='utf-8') as f:
        judge_results = json.load(f)
    with open(args.rewritten_rollouts_json, 'r', encoding='utf-8') as f:
        rewritten_rollouts = json.load(f)
    
    print(f"Loaded {len(judge_results)} judge results")
    print(f"Loaded {len(rewritten_rollouts)} rewritten rollouts")
    
    # 创建数据集
    dataset = RewriteRolloutDataset(
        judge_results=judge_results,
        rewritten_rollouts=rewritten_rollouts,
        data_path=args.data_path,
        media_root=args.media_root,
        data_type=args.data_type,
        score_threshold=args.score_threshold,
        n_samples=args.n_samples
    )
    
    if len(dataset) == 0:
        print("Error: No matched samples found!")
        return
    
    # 获取模型wrapper
    wrapper = model_to_wrapper_map[args.model_name][args.model_size]
    
    # 生成并保存向量
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
        track_cost=args.track_cost,
    )


if __name__ == "__main__":
    main()
