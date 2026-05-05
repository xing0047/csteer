"""
为 Refer VPT vs Raw 生成 steering vectors。
通过对比有数字标注的图像（images_vpt）和原图（images_raw）的激活差异来生成向量。

Example usage:
python generate_vector_refer_vpt.py \
    --model_name qwen3vl \
    --model_size 8b \
    --layers $(seq 0 35) \
    --behavior_paths ../PAIRS/refer_vpt_exp/LIUH0115_qwen3vl_8b_inst_it_image_refer_vpt_n1024 \
    --output_dir refer_vpt_exp \
    --use_flash_attn \
    --verbose
"""

import torch as t
from torch.utils.data import Dataset
from tqdm import tqdm
import os
import argparse
from typing import List
from behaviors import (
    get_vector_dir,
    get_vector_path,
    REFER_VPT,
)
from datasets import load_from_disk
from utils.mwrapper_map import (
    model_to_wrapper_map
)
from utils.conversation import (
    INST_IT_IMAGE_SYSTEM_PROMPT,
    INST_IT_IMAGE_SYSTEM_PROMPT_NO_ID,
    INST_IT_VIDEO_SYSTEM_PROMPT,
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, choices=["internvl3", "internvl3_5", "qwen3vl"]
    )
    parser.add_argument(
        "--model_size", type=str, choices=["2b", "8b", "32b", "38b"]
    )
    parser.add_argument(
        "--layers", nargs="+", type=int, default=list(range(28))
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
    parser.add_argument(
        "--behavior_paths", nargs="+", type=str, required=True
    )
    parser.add_argument(
        "--output_dir", type=str, required=True
    )
    return parser.parse_args()


class ReferVPTDataset(Dataset):
    """用于 Refer VPT vs Raw 对比的数据集类"""
    def __init__(self, wrapper, data_path, model_name):
        self.data = load_from_disk(data_path)
        self.wrapper = wrapper
        self.model = wrapper.model
        if "inst_it_image" in data_path:
            self.data_name = "inst_it_image"
            self.vision_type = "image"
        elif "inst_it_video" in data_path:
            self.data_name = "inst_it_video"
            self.vision_type = "video"
        else:
            raise ValueError(f"Unsupported data type in path: {data_path}")
        self.model_name = model_name
        self.tokenizer = wrapper.tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def prompt_to_tokens(self, image_or_video_path, vision_type, instruction, system_prompt=None, verbose=False):
        """将提示词转换为token，不包含模型输出（因为我们只提取激活，不生成）"""
        tokens = self.wrapper.get_tokens_for_compare(
            image_or_video_path, 
            vision_type, 
            instruction, 
            model_output="",  # 空输出，因为我们只需要提取激活
            system_prompt=system_prompt,  # 允许指定自定义 system prompt
            verbose=verbose
        )
        return t.tensor(tokens).unsqueeze(0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        if self.data_name == "inst_it_image":
            image_vpt_path = item["image_vpt"]  # 有数字标注的图像
            image_raw_path = item["image_raw"]  # 原图
            instruction_vpt = item["instruction_vpt"]  # 有标注图的指令（要求使用IDs）
            instruction_raw = item["instruction_raw"]  # 原图的指令（不要求使用IDs）
            
            # 为VPT和Raw分别生成token
            # VPT: 使用包含ID规则的 system prompt
            # Raw: 使用不包含ID规则的 system prompt
            tokens_vpt = self.prompt_to_tokens(
                image_vpt_path, "image", instruction_vpt, 
                system_prompt=INST_IT_IMAGE_SYSTEM_PROMPT,  # 有ID规则
                verbose=False
            )
            tokens_raw = self.prompt_to_tokens(
                image_raw_path, "image", instruction_raw,
                system_prompt=INST_IT_IMAGE_SYSTEM_PROMPT_NO_ID,  # 无ID规则
                verbose=False
            )
            return image_vpt_path, image_raw_path, "image", tokens_vpt, tokens_raw, instruction_vpt, instruction_raw
        elif self.data_name == "inst_it_video":
            video_vpt_path = item["video_vpt"]
            video_raw_path = item["video_raw"]
            instruction_vpt = item["instruction_vpt"]
            instruction_raw = item["instruction_raw"]
            tokens_vpt = self.prompt_to_tokens(video_vpt_path, "video", instruction_vpt, verbose=False)
            tokens_raw = self.prompt_to_tokens(video_raw_path, "video", instruction_raw, verbose=False)
            return video_vpt_path, video_raw_path, "video", tokens_vpt, tokens_raw, instruction_vpt, instruction_raw
        else:
            raise NotImplementedError


def generate_save_vectors_for_behavior(
    layers: List[int],
    model_name: str,
    model_size: str,
    behavior: str,
    behavior_path: str,
    model: None,
    output_dir: str,
    verbose: bool = False,
):
    """生成并保存 Refer VPT vs Raw 的 steering vectors"""
    if not os.path.exists(get_vector_dir(behavior, model_name, model_size, output_dir)):
        os.makedirs(get_vector_dir(behavior, model_name, model_size, output_dir))

    model.set_save_internal_decodings(False)
    model.reset_all()

    vpt_activations = dict([(layer, []) for layer in layers])  # 有标注图的激活
    raw_activations = dict([(layer, []) for layer in layers])  # 原图的激活

    dataset = ReferVPTDataset(
        model,
        behavior_path,
        model_name,
    )

    for vpt_path, raw_path, vision_type, tokens_vpt, tokens_raw, instruction_vpt, instruction_raw in tqdm(dataset, desc="Processing image pairs"):
        tokens_vpt = tokens_vpt.to(model.device)
        tokens_raw = tokens_raw.to(model.device)
        
        # 处理有标注图（VPT）- 使用要求IDs的指令
        model.reset_all()
        model.get_logits(vpt_path, vision_type, tokens_vpt)
        for layer in layers:
            vpt_act = model.get_last_activations(layer)
            # 处理可能的3维激活 (batch, seq_len, hidden_dim) -> (seq_len, hidden_dim)
            if len(vpt_act.shape) == 3:
                vpt_act = vpt_act.squeeze(0)  # 移除batch维度
            assert len(vpt_act.shape) == 2, f"Expected 2D activations, got shape {vpt_act.shape}"
            # 取倒数第二个token的激活（通常是答案生成前的关键位置）
            vpt_act = vpt_act[-2, :].detach().cpu()
            vpt_activations[layer].append(vpt_act)
        
        # 处理原图（Raw）- 使用不要求IDs的指令
        model.reset_all()
        model.get_logits(raw_path, vision_type, tokens_raw)
        for layer in layers:
            raw_act = model.get_last_activations(layer)
            # 处理可能的3维激活 (batch, seq_len, hidden_dim) -> (seq_len, hidden_dim)
            if len(raw_act.shape) == 3:
                raw_act = raw_act.squeeze(0)  # 移除batch维度
            assert len(raw_act.shape) == 2, f"Expected 2D activations, got shape {raw_act.shape}"
            raw_act = raw_act[-2, :].detach().cpu()
            raw_activations[layer].append(raw_act)
        
        if verbose:
            print(f"[VPT Image]: {vpt_path}")
            print(f"[Raw Image]: {raw_path}")
            print(f"[Instruction VPT]: {instruction_vpt}")
            print(f"[Instruction Raw]: {instruction_raw}")
            print("-" * 80)

    # 计算并保存向量
    for layer in layers:
        all_vpt_layer = t.stack(vpt_activations[layer])
        all_raw_layer = t.stack(raw_activations[layer])
        # Steering vector = mean(VPT激活 - Raw激活)
        # 这表示"有标注图相对于原图"的激活方向
        vec = (all_vpt_layer - all_raw_layer).mean(dim=0)
        t.save(
            vec,
            get_vector_path(behavior, model_name, model_size, output_dir, layer),
        )
    
    print(f"Vectors saved locally to: {get_vector_dir(behavior, model_name, model_size, output_dir)}")


def generate_save_vectors(
    layers: List[int],
    wrapper: str,
    model_name: str,
    model_size: str,
    use_flash_attn: bool,
    behavior_paths: List[str],
    output_dir: str,
    verbose: bool = False,
):
    """
    生成并保存 Refer VPT vs Raw 的 steering vectors
    
    Args:
        layers: 要生成向量的层列表
        wrapper: 模型包装器
        model_name: 模型名称
        model_size: 模型大小
        use_flash_attn: 是否使用flash attention
        behavior_paths: 行为数据路径列表（对于refer_vpt，通常只有一个）
        output_dir: 输出目录
        verbose: 是否输出详细信息
    """
    model = wrapper(
        name=model_name,
        size=model_size,
        use_flash_attn=use_flash_attn
    )
    
    # 对于 refer_vpt，我们使用固定的 behavior 名称
    behavior = REFER_VPT
    
    # 通常只有一个 behavior_path
    for behavior_path in behavior_paths:
        generate_save_vectors_for_behavior(
            layers, model_name, model_size, behavior, 
            behavior_path, model, output_dir, verbose
        )


if __name__ == "__main__":
    args = parse_args()
    wrapper = model_to_wrapper_map[args.model_name][args.model_size]
    generate_save_vectors(
        args.layers,
        wrapper,
        args.model_name,
        args.model_size,
        args.use_flash_attn,
        args.behavior_paths,
        args.output_dir,
        args.verbose
    )

