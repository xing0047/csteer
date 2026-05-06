"""
Generate comparison pairs for Refer VPT vs Raw.
Pairs compare annotated images (images_vpt) with the corresponding raw images (images_raw).

Example usage:
python generate_refer_vpt.py \
    --model_name qwen3vl \
    --model_size 8b \
    --data inst_it_image \
    --n_pairs 1024 \
    --output_dir ../PAIRS/refer_vpt_exp \
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

    # Generation Config
    generation_config = dict(
        max_new_tokens=1024, do_sample=False, repetition_penalty=1.0
    )

    # Load Model
    if args.model_name == 'internvl3' or args.model_name == 'internvl3_5':
        if args.model_size == '2b' or args.model_size == '8b':
            model = AutoModel.from_pretrained(
                path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                use_flash_attn=args.use_flash_attn,
                trust_remote_code=True).eval().cuda()
        elif args.model_size == '38b':
            import math
            from transformers import AutoConfig
            def split_model(model_name):
                device_map = {}
                world_size = torch.cuda.device_count()
                config = AutoConfig.from_pretrained(path, trust_remote_code=True)
                num_layers = config.llm_config.num_hidden_layers
                # Since the first GPU will be used for ViT, treat it as half a GPU.
                num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
                num_layers_per_gpu = [num_layers_per_gpu] * world_size
                num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
                layer_cnt = 0
                for i, num_layer in enumerate(num_layers_per_gpu):
                    for j in range(num_layer):
                        device_map[f'language_model.model.layers.{layer_cnt}'] = i
                        layer_cnt += 1
                device_map['vision_model'] = 0
                device_map['mlp1'] = 0
                device_map['language_model.model.tok_embeddings'] = 0
                device_map['language_model.model.embed_tokens'] = 0
                device_map['language_model.output'] = 0
                device_map['language_model.model.norm'] = 0
                device_map['language_model.model.rotary_emb'] = 0
                device_map['language_model.lm_head'] = 0
                device_map[f'language_model.model.layers.{num_layers - 1}'] = 0
                return device_map
            device_map = split_model('InternVL3-38B')
            model = AutoModel.from_pretrained(
                path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                use_flash_attn=args.use_flash_attn,
                trust_remote_code=True,
                device_map=device_map).eval()
        else:
            raise ValueError(f'No {args.model_name}_{args.model_size}')
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    elif args.model_name == 'qwen3vl':
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            path, 
            torch_dtype="auto", 
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2" if args.use_flash_attn else "eager", 
            device_map="auto"
        ).eval().cuda()
        processor = AutoProcessor.from_pretrained(
            path
        )
        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)
    else:
        raise ValueError(f'No {args.model_name}_{args.model_size}')

    # Load Data
    if args.data == "inst_it_image":
        img_root = "../DATA/Inst-It-Dataset"
        img_vpt_root = "../DATA/Inst-It-Dataset/images_vpt"
        img_raw_root = "../DATA/Inst-It-Dataset/images_raw"
        ann_path = "../DATA/Inst-It-Dataset/inst_it_dataset_image_51k.json"
        data = json.load(open(ann_path))[:args.n_pairs]
        data_dict = {
            'image_vpt': [],  # annotated image path (with numeric IDs)
            'image_raw': [],  # raw image path
            'caption_gt': [],  # ground-truth caption (may be used for later evaluation)
            'instruction_vpt': [],  # instruction for annotated image (requires IDs)
            'instruction_raw': [],  # instruction for raw image (does not require IDs)
        }
    elif args.data == "inst_it_video":
        vid_root = "../DATA/Inst-It-Dataset"
        ann_path = "../DATA/Inst-It-Dataset/inst_it_dataset_video_21k.json"
        data = json.load(open(ann_path))[:args.n_pairs]
        data_dict = {
            'video_vpt': [],
            'video_raw': [],
            'caption_gt': [],
            'instruction_vpt': [],
            'instruction_raw': [],
        }
    else:
        raise ValueError(f'No {args.data}')
    
    # Generate contrastive pairs
    # Annotated image: requires describing with IDs
    question_vpt = "Please describe the whole image with IDs."
    # Raw image: does not require IDs (raw images have no numeric annotations)
    question_raw = "Please describe the whole image."
    
    for item in tqdm(data, desc="Processing images"):
        if args.data == "inst_it_image":
            # Get original image path (may already include an images_vpt/ prefix)
            original_image_path = item['image_path']
            
            # Extract relative path (strip images_vpt/ or images_raw/ prefix)
            if original_image_path.startswith('images_vpt/'):
                relative_path = original_image_path[len('images_vpt/'):]
            elif original_image_path.startswith('images_raw/'):
                relative_path = original_image_path[len('images_raw/'):]
            else:
                relative_path = original_image_path
            
            # Build full paths for annotated and raw images
            image_vpt_path = os.path.join(img_vpt_root, relative_path)
            image_raw_path = os.path.join(img_raw_root, relative_path)
            
            # Check files exist
            if not os.path.exists(image_vpt_path):
                if args.verbose:
                    print(f"Warning: VPT image not found: {image_vpt_path}, skipping...")
                continue
            if not os.path.exists(image_raw_path):
                if args.verbose:
                    print(f"Warning: Raw image not found: {image_raw_path}, skipping...")
                continue
            
            caption_gt = item["image_level_caption"].strip()
            
            data_dict["image_vpt"].append(image_vpt_path)
            data_dict["image_raw"].append(image_raw_path)
            data_dict["caption_gt"].append(caption_gt)
            data_dict["instruction_vpt"].append(question_vpt)
            data_dict["instruction_raw"].append(question_raw)
            
            if args.verbose:
                print(f"[VPT]: {image_vpt_path}")
                print(f"[Raw]: {image_raw_path}")
                print(f"[GT]: {caption_gt}")
        elif args.data == "inst_it_video":
            # Video handling (if needed)
            raise NotImplementedError("Video support for refer_vpt is not implemented yet")
        else:
            raise ValueError(f'No {args.data}')

    # Save
    ds = Dataset.from_dict(data_dict)
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, f"{args.hf_user}_{args.model_name}_{args.model_size}_{args.data}_refer_vpt_n{len(data_dict['image_vpt'])}")
    ds.save_to_disk(save_path)
    print(f"Dataset saved locally to: {save_path}")
    print(f"Total pairs: {len(data_dict['image_vpt'])}")


if __name__ == "__main__":
    main()

