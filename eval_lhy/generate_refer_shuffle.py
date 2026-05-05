"""
生成 Exact Matching vs Prompt Shuffle 的对比数据对。
对比正确匹配的图像（images_vpt）和打乱匹配的图像（images_shuffle）的激活差异。

Example usage:
python generate_refer_shuffle.py \
    --model_name qwen3vl \
    --model_size 8b \
    --data inst_it_image \
    --n_pairs 1024 \
    --output_dir ../PAIRS/refer_shuffle_exp \
    --hf_user LIUH0115 \
    --use_flash_attn \
    --verbose
"""

import argparse
import torch
import os
import json
from transformers import (
    AutoModel, 
    AutoTokenizer,
    AutoProcessor,
    Qwen3VLForConditionalGeneration
)
from PIL import Image
from datasets import Dataset
from tqdm import tqdm
from utils.load_image import (
    internvl_load_image,
    internvl_load_frames,    
)
from utils.mpath_map import model_to_path_map
from utils.conversation import (
    INST_IT_IMAGE_SYSTEM_PROMPT,
    INST_IT_VIDEO_SYSTEM_PROMPT,
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name', required=True, choices=['internvl3', 'internvl3_5', 'qwen3vl']
    )
    parser.add_argument(
        '--model_size', required=True, choices=['2b', '8b', '32b', '38b']
    )
    parser.add_argument(
        '--data', type=str, default='inst_it_image', choices=['inst_it_image', 'inst_it_video']
    )
    parser.add_argument(
        '--n_pairs', type=int, default=1024
    )
    parser.add_argument(
        '--output_dir', type=str, default='contrastive_pairs'
    )
    parser.add_argument(
        '--hf_user', type=str, default="LIUH0115", choices=["LIUH0115", "xing0047"]
    )
    parser.add_argument(
        '--use_flash_attn', action='store_true', help="enable flash attention"
    )
    parser.add_argument(
        '--use_vllm', action='store_true', help="enable vllm inference"
    )
    parser.add_argument(
        '--verbose', action='store_true', help="enable verbose outputs"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    path = model_to_path_map[args.model_name][args.model_size]

    # Load Data
    if args.data == "inst_it_image":
        img_root = "../DATA/Inst-It-Dataset"
        img_vpt_root = "../DATA/Inst-It-Dataset/images_vpt"
        img_shuffle_root = "../DATA/Inst-It-Dataset/images_shuffle"
        # 使用带mask的标注文件，因为shuffle图像是基于这个顺序的
        ann_path = "../DATA/Inst-It-Dataset/inst_it_dataset_image_51k.json"
        data = json.load(open(ann_path))[:args.n_pairs]
        data_dict = {
            'image_vpt': [],  # 正确匹配的图像路径（images_vpt）
            'image_shuffle': [],  # 打乱匹配的图像路径（images_shuffle）
            'caption_gt': [],  # 真实描述（用于后续可能的评估）
            'instruction': [],  # 指令文本（两个都用相同的指令，要求使用IDs）
        }
    elif args.data == "inst_it_video":
        vid_root = "../DATA/Inst-It-Dataset"
        ann_path = "../DATA/Inst-It-Dataset/inst_it_dataset_video_21k.json"
        data = json.load(open(ann_path))[:args.n_pairs]
        data_dict = {
            'video_vpt': [],
            'video_shuffle': [],
            'caption_gt': [],
            'instruction': [],
        }
    else:
        raise ValueError(f'No {args.data}')
    
    # Generate Contrastive Pairs
    # 两个都使用相同的指令，要求使用IDs（因为都有ID框）
    question = "Please describe the whole image with IDs."
    
    for item in tqdm(data, desc="Processing images"):
        if args.data == "inst_it_image":
            # 获取原始图像路径（相对于 Inst-It-Dataset 根目录）
            original_image_path = item['image_path']
            
            # 从路径中提取相对路径（去掉 images_vpt/ 或 images_shuffle/ 前缀）
            if original_image_path.startswith('images_vpt/'):
                relative_path = original_image_path[len('images_vpt/'):]
            elif original_image_path.startswith('images_shuffle/'):
                relative_path = original_image_path[len('images_shuffle/'):]
            else:
                relative_path = original_image_path
            
            # 构建正确匹配图和打乱匹配图的完整路径
            image_vpt_path = os.path.join(img_vpt_root, relative_path)
            image_shuffle_path = os.path.join(img_shuffle_root, relative_path)
            
            # 检查文件是否存在
            if not os.path.exists(image_vpt_path):
                if args.verbose:
                    print(f"Warning: VPT image not found: {image_vpt_path}, skipping...")
                continue
            if not os.path.exists(image_shuffle_path):
                if args.verbose:
                    print(f"Warning: Shuffle image not found: {image_shuffle_path}, skipping...")
                continue
            
            caption_gt = item["image_level_caption"].strip()
            
            data_dict["image_vpt"].append(image_vpt_path)
            data_dict["image_shuffle"].append(image_shuffle_path)
            data_dict["caption_gt"].append(caption_gt)
            data_dict["instruction"].append(question)
            
            if args.verbose:
                print(f"[VPT (Exact Match)]: {image_vpt_path}")
                print(f"[Shuffle (False Match)]: {image_shuffle_path}")
                print(f"[GT]: {caption_gt}")
        elif args.data == "inst_it_video":
            # 视频处理逻辑（如果需要）
            raise NotImplementedError("Video support for refer_shuffle is not implemented yet")
        else:
            raise ValueError(f'No {args.data}')

    # Save
    ds = Dataset.from_dict(data_dict)
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, f"{args.hf_user}_{args.model_name}_{args.model_size}_{args.data}_refer_shuffle_n{len(data_dict['image_vpt'])}")
    ds.save_to_disk(save_path)
    print(f"Dataset saved locally to: {save_path}")
    print(f"Total pairs: {len(data_dict['image_vpt'])}")


if __name__ == "__main__":
    main()
