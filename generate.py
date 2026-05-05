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
from datasets import load_dataset, Dataset
from tqdm import tqdm
from utils.internvl_utils import (
    load_image as internvl_load_image,
    load_frames as internvl_load_frames,
)
from utils.helpers import get_model_path
from utils.conversation import (
    INST_IT_IMAGE_SYSTEM_PROMPT,
    INST_IT_VIDEO_SYSTEM_PROMPT,
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name', required=True, choices=['internvl3_5', 'qwen3vl']
    )
    parser.add_argument(
        '--data', type=str, default='inst_it_image', choices=['refcocog', 'inst_it_image', 'inst_it_video']
    )
    parser.add_argument(
        '--n_pairs', type=int, default=1024
    )
    parser.add_argument(
        '--output_dir', type=str, default='contrastive_pairs'
    )
    parser.add_argument(
        '--hf_user', type=str, default="LIUH0115", choices=["LIUH0015", "xing0047"]
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
    path = get_model_path(args.model_name, "8b")

    # Generation Config
    generation_config = dict(
        max_new_tokens=1024, do_sample=False, repetition_penalty=1.0
    )

    # Load Model
    if args.model_name == 'internvl3_5':
        model = AutoModel.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=args.use_flash_attn,
            trust_remote_code=True).eval().cuda()
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
        raise ValueError(f'Unsupported model_name {args.model_name}')

    # Load Data
    if args.data == "refcocog":
        raise NotImplementedError
        data = load_dataset('../DATA/refcocog')['train'][:args.n_pairs]
    elif args.data == "inst_it_image":
        img_root = "../DATA/Inst-It-Dataset"
        ann_path = "../DATA/Inst-It-Dataset/inst_it_dataset_image_51k.json"
        data = json.load(open(ann_path))[:args.n_pairs]
        data_dict = {
            'image': [], 
            'caption_gt': [], 
            'caption_generated': [], 
            'answer_matching_behavior': [], 
            'answer_not_matching_behavior': []
        }
    elif args.data == "inst_it_video":
        vid_root = "../DATA/Inst-It-Dataset"
        ann_path = "../DATA/Inst-It-Dataset/inst_it_dataset_video_21k.json"
        data = json.load(open(ann_path))[:args.n_pairs]
        data_dict = {
            'video': [], 
            'caption_gt': [], 
            'caption_generated': [], 
            'answer_matching_behavior': [], 
            'answer_not_matching_behavior': []
        }
    else:
        raise ValueError(f'No {args.data}')
    
    # Generate Contrastive Pairs
    for item in tqdm(data):
        if args.data == "refcocog":
            raise NotImplementedError
            image_path = f"../data/{item['image_path']}"
            assert isinstance(item['bbox'][0], float)
            pixel_values = load_image(image_path, max_num=12, box=item['bbox'], box_color='red').to(torch.bfloat16).cuda()
            question = '<image>\nPlease describe the object in red box shortly.'
            caption_gt = f"The object in red box is a {item['sentences'][-1]['sent']}"
        elif args.data == "inst_it_image":
            image_path = f"{img_root}/{item['image_path']}"
            caption_gt = item["image_level_caption"].strip()
        elif args.data == "inst_it_video":
            frame_names = [f"{item['video_path']}/{frame_info['frame_name']}" for frame_info in item["frame_level_caption"]]
            video_path = [os.path.join(vid_root, frame_name) for frame_name in frame_names]
            question = "Please briefly describe the whole video with IDs."
            caption_gt = item["video_level_caption"].strip()
        else:
            raise ValueError(f'No {args.data}')
        
        if args.model_name == 'internvl3_5':
            if args.data == "inst_it_image":
                pixel_values = internvl_load_image(image_path).to(torch.bfloat16).cuda()
                question = INST_IT_IMAGE_SYSTEM_PROMPT.strip() + "\n<image>\nPlease describe the whole image with IDs."
                response = model.chat(tokenizer, pixel_values, question, generation_config)
                data_dict["image"].append(image_path)
            elif args.data == "inst_it_video":
                pixel_values, num_patches_list = internvl_load_frames(video_path)
                question = INST_IT_VIDEO_SYSTEM_PROMPT.strip() + '\n' + ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))]) + "Please describe the whole video with IDs."
                response = model.chat(tokenizer, pixel_values, question, generation_config, num_patches_list=num_patches_list)
                data_dict["video"].append(video_path)
            else:
                raise NotImplementedError
        elif args.model_name == 'qwen3vl':
            if args.data == "inst_it_image":
                question = "Please describe the whole image with IDs."
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": INST_IT_IMAGE_SYSTEM_PROMPT},
                            {"type": "image", "image": image_path},
                            {"type": "text", "text": question},
                        ],
                    }
                ]
                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                data_dict['image'].append(image_path)
            elif args.data == "inst_it_video":
                question = "Please describe the whole video with IDs."
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": INST_IT_VIDEO_SYSTEM_PROMPT},
                            {"type": "video", "video": video_path},
                            {"type": "text", "text": question},
                        ],
                    }
                ]
                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                    do_sample_frames=False
                )
                data_dict['video'].append(video_path)
            else:
                raise NotImplementedError
            inputs = inputs.to(model.device)
            generated = model.generate(**inputs, **generation_config)
            generated_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated)
            ]
            response = tokenizer.batch_decode(generated_trimmed, skip_special_tokens=True)[0]
        else:
            raise NotImplementedError
        if args.verbose:
            print(f"[q]: \033[31m{question}\033[0m")
            print(f"[a]: \033[93m{response}\033[0m")
        
        data_dict['caption_gt'].append(caption_gt)
        data_dict['caption_generated'].append(response)
        data_dict['answer_matching_behavior'].append('(A)')
        data_dict['answer_not_matching_behavior'].append('(B)')

    # Save
    ds = Dataset.from_dict(data_dict)
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, f"{args.hf_user}_{args.model_name}_8b_{args.data}_n{args.n_pairs}")
    ds.save_to_disk(save_path)
    print(f"Dataset saved locally to: {save_path}")


if __name__ == "__main__":
    main()