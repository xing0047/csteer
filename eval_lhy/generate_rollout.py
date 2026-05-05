"""
为视频生成rollout并打分
通过vLLM生成多个rollout，然后使用judge模型判断分数，保存结果到JSON

支持的模型：internvl3, internvl3_5, qwen3vl

支持两种模式：
1. image: 图像模式（原有功能）
2. video: 视频模式（新增功能）

Example usage (video with internvl3_5):
python generate_rollout.py \
    --model_name internvl3_5 \
    --model_size 8b \
    --data_type video \
    --data_path ../DATA/Inst-It-Dataset/inst_it_dataset_video_21k.json \
    --media_root ../DATA/Inst-It-Dataset \
    --n_samples 1024 \
    --num_rollouts 8 \
    --rollout_temperature 0.6 \
    --judge_base_url http://localhost:8001/v1 \
    --judge_model_name Qwen/Qwen2.5-72B-Instruct-AWQ \
    --output_dir video_rollout_exp \
    --verbose

Example usage (video with qwen3vl):
python generate_rollout.py \
    --model_name qwen3vl \
    --model_size 8b \
    --data_type video \
    --data_path ../DATA/Inst-It-Dataset/inst_it_dataset_video_21k.json \
    --media_root ../DATA/Inst-It-Dataset \
    --n_samples 1024 \
    --num_rollouts 8 \
    --rollout_temperature 0.6 \
    --judge_base_url http://localhost:8001/v1 \
    --judge_model_name Qwen/Qwen2.5-72B-Instruct-AWQ \
    --output_dir video_rollout_exp \
    --verbose

Example usage (image):
python generate_rollout.py \
    --model_name internvl3_5 \
    --model_size 8b \
    --data_type image \
    --data_path ../DATA/Inst-It-Dataset/inst_it_dataset_image_51k.json \
    --media_root ../DATA/Inst-It-Dataset \
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
import time
import psutil


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
    parser.add_argument(
        '--track_cost', action='store_true',
        help="Track and save cost metrics (time, tokens, etc.) to a JSON file"
    )
    return parser.parse_args()


class VideoDataset(Dataset):
    """用于视频数据的数据集类"""
    def __init__(self, data_path, n_samples, media_root):
        with open(data_path, 'r', encoding='utf-8') as f:
            all_data = json.load(f)
        self.data = all_data[:n_samples]
        self.media_root = media_root

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # 获取视频帧路径列表
        video_path = item['video_path']  # e.g., "videos_vpt/YoutubeVIS-2021/train/05e0b0f28f"
        frame_names = [frame_info['frame_name'] for frame_info in item['frame_level_caption']]
        
        # 构建完整的帧路径列表
        frame_paths = [os.path.join(self.media_root, video_path, frame_name) for frame_name in frame_names]
        
        caption_gt = item["video_level_caption"].strip()
        instruction = "Please describe the whole video with IDs."
        
        # 提取视频ID作为名称
        video_name = os.path.basename(video_path)
        
        return frame_paths, instruction, caption_gt, video_name, idx


class ImageDataset(Dataset):
    """用于图像数据的数据集类"""
    def __init__(self, data_path, n_samples, media_root):
        with open(data_path, 'r', encoding='utf-8') as f:
            all_data = json.load(f)
        self.data = all_data[:n_samples]
        self.media_root = media_root

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # 获取图像路径
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
    使用vLLM为视频生成多个rollout
    
    Args:
        vllm_llm: vLLM LLM实例
        frame_paths: 视频帧路径列表
        instruction: 指令
        system_prompt: 系统提示词
        num_rollouts: 生成rollout的数量
        temperature: 采样温度
        max_tokens: 最大生成token数
    
    Returns:
        List[str]: 生成的rollout列表
    """
    full_prompt = f"{system_prompt}\n\n{instruction}"
    
    # 构建多图像消息内容
    content = []
    
    # 添加每个视频帧作为图像
    for i, frame_path in enumerate(frame_paths):
        if not os.path.isabs(frame_path):
            frame_path = os.path.abspath(frame_path)
        
        if not os.path.exists(frame_path):
            print(f"Warning: Frame not found: {frame_path}")
            continue
        
        # 使用 file:// URL 格式
        frame_url = f"file://{frame_path}"
        content.append({
            "type": "image_url",
            "image_url": {"url": frame_url},
        })
    
    # 添加文本提示
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
    使用vLLM为图像生成多个rollout
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
    """从judge模型的输出中提取分数（0-1之间）"""
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
    使用judge模型判断rollout的分数
    """
    # 根据数据类型选择prompt模板
    if "gpt_eval_rollout" in prompt_template and data_type in prompt_template["gpt_eval_rollout"]:
        eval_prompt_template = prompt_template["gpt_eval_rollout"][data_type]
    elif "gpt_eval" in prompt_template and data_type in prompt_template["gpt_eval"]:
        eval_prompt_template = prompt_template["gpt_eval"][data_type]
    else:
        # 默认使用image的模板
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


def main():
    args = parse_args()
    
    cost_data = {}
    t_total_start = time.time()
    
    # 设置输出目录
    output_dir = os.path.join(args.output_root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取模型路径
    if args.vllm_model_path is None:
        vllm_model_path = model_to_path_map[args.model_name][args.model_size]
    else:
        vllm_model_path = args.vllm_model_path
    
    # 确定允许的本地媒体路径
    media_root_abs = os.path.abspath(args.media_root)
    
    print("=" * 50)
    print(f"生成Rollout并打分")
    print("=" * 50)
    print(f"数据类型: {args.data_type}")
    print(f"模型: {args.model_name}_{args.model_size}")
    print(f"数据路径: {args.data_path}")
    print(f"媒体根目录: {media_root_abs}")
    print(f"样本数: {args.n_samples}")
    print(f"Rollout数量: {args.num_rollouts}")
    print(f"输出目录: {output_dir}")
    if args.track_cost:
        print(f"开销统计: 已启用")
    print("=" * 50)
    
    # 加载vLLM模型
    print(f"\n加载vLLM模型: {vllm_model_path}")
    t_model_load_start = time.time()
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
    t_model_load_end = time.time()
    print("vLLM模型加载完成")
    
    if args.track_cost:
        cost_data["model_load_time_s"] = round(t_model_load_end - t_model_load_start, 3)
        cost_data["gpu_after_model_load"] = _get_gpu_memory_info()
    
    # 连接judge模型服务
    judge_client = OpenAI(base_url=args.judge_base_url, api_key="EMPTY", timeout=600)
    print(f"已连接Judge服务: {args.judge_base_url}")
    
    # 创建数据集
    if args.data_type == "video":
        dataset = VideoDataset(args.data_path, args.n_samples, media_root_abs)
        system_prompt = INST_IT_VIDEO_SYSTEM_PROMPT
    else:
        dataset = ImageDataset(args.data_path, args.n_samples, media_root_abs)
        system_prompt = INST_IT_IMAGE_SYSTEM_PROMPT
    
    print(f"数据集大小: {len(dataset)}")
    
    # 存储结果
    all_results = []
    
    # 开销统计变量
    rollout_gen_times = []
    judge_times = []
    total_rollout_tokens = 0
    skipped_samples = 0
    
    t_inference_start = time.time()
    
    for idx in tqdm(range(len(dataset)), desc="Processing samples"):
        if args.data_type == "video":
            frame_paths, instruction, caption_gt, media_name, sample_idx = dataset[idx]
            
            # 检查帧是否存在
            valid_frames = [p for p in frame_paths if os.path.exists(p)]
            if len(valid_frames) == 0:
                if args.verbose:
                    print(f"Warning: No valid frames for video {media_name}, skipping...")
                skipped_samples += 1
                continue
            
            # 生成rollouts
            t_gen_start = time.time()
            rollouts = generate_rollouts_for_video(
                vllm_llm=vllm_llm,
                frame_paths=valid_frames,
                instruction=instruction,
                system_prompt=system_prompt,
                num_rollouts=args.num_rollouts,
                temperature=args.rollout_temperature,
            )
            t_gen_end = time.time()
        else:
            image_path, instruction, caption_gt, media_name, sample_idx = dataset[idx]
            
            if not os.path.exists(image_path):
                if args.verbose:
                    print(f"Warning: Image not found: {image_path}, skipping...")
                skipped_samples += 1
                continue
            
            t_gen_start = time.time()
            rollouts = generate_rollouts_for_image(
                vllm_llm=vllm_llm,
                image_path=image_path,
                instruction=instruction,
                system_prompt=system_prompt,
                num_rollouts=args.num_rollouts,
                temperature=args.rollout_temperature,
            )
            t_gen_end = time.time()
        
        if args.track_cost:
            rollout_gen_times.append(t_gen_end - t_gen_start)
            for r in rollouts:
                total_rollout_tokens += len(r.split()) if r else 0
        
        # 打分
        rollout_details = []
        t_judge_start = time.time()
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
        t_judge_end = time.time()
        
        if args.track_cost:
            judge_times.append(t_judge_end - t_judge_start)
        
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
    
    t_inference_end = time.time()
    
    # 保存JSON结果
    json_path = os.path.join(output_dir, "judge_results.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\nJSON结果已保存到: {json_path}")
    
    # 保存CSV结果
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
    
    print(f"CSV结果已保存到: {csv_path}")
    print(f"总样本数: {len(all_results)}")
    
    # 统计信息
    total_rollouts = len(all_results) * args.num_rollouts
    total_incorrect = sum(r['num_incorrect'] for r in all_results)
    print(f"总Rollout数: {total_rollouts}")
    if total_rollouts > 0:
        print(f"错误Rollout数: {total_incorrect} ({100*total_incorrect/total_rollouts:.1f}%)")
    
    t_total_end = time.time()
    
    # 保存开销统计
    if args.track_cost:
        cost_data["step"] = "generate_rollout"
        cost_data["model"] = f"{args.model_name}_{args.model_size}"
        cost_data["data_type"] = args.data_type
        cost_data["n_samples_requested"] = args.n_samples
        cost_data["n_samples_processed"] = len(all_results)
        cost_data["n_samples_skipped"] = skipped_samples
        cost_data["num_rollouts_per_sample"] = args.num_rollouts
        cost_data["total_rollouts_generated"] = total_rollouts
        cost_data["total_wall_time_s"] = round(t_total_end - t_total_start, 3)
        cost_data["inference_time_s"] = round(t_inference_end - t_inference_start, 3)
        cost_data["rollout_generation"] = {
            "total_time_s": round(sum(rollout_gen_times), 3),
            "avg_time_per_sample_s": round(sum(rollout_gen_times) / len(rollout_gen_times), 3) if rollout_gen_times else 0,
            "min_time_s": round(min(rollout_gen_times), 3) if rollout_gen_times else 0,
            "max_time_s": round(max(rollout_gen_times), 3) if rollout_gen_times else 0,
        }
        cost_data["judge_scoring"] = {
            "total_time_s": round(sum(judge_times), 3),
            "avg_time_per_sample_s": round(sum(judge_times) / len(judge_times), 3) if judge_times else 0,
            "min_time_s": round(min(judge_times), 3) if judge_times else 0,
            "max_time_s": round(max(judge_times), 3) if judge_times else 0,
        }
        cost_data["approx_rollout_tokens_generated"] = total_rollout_tokens
        cost_data["gpu_after_inference"] = _get_gpu_memory_info()
        cost_data["system"] = {
            "cpu_count": psutil.cpu_count(),
            "ram_total_GB": round(psutil.virtual_memory().total / 1024**3, 2),
            "ram_used_GB": round(psutil.virtual_memory().used / 1024**3, 2),
        }
        
        cost_json_path = os.path.join(output_dir, "cost_generate_rollout.json")
        with open(cost_json_path, 'w', encoding='utf-8') as f:
            json.dump(cost_data, f, ensure_ascii=False, indent=2)
        print(f"\n开销统计已保存到: {cost_json_path}")


if __name__ == "__main__":
    main()
