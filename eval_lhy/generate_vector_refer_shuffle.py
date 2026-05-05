"""
为 Exact Matching vs Prompt Shuffle 生成 steering vectors。
通过对比正确匹配的图像（images_vpt）和打乱匹配的图像（images_shuffle）的激活差异来生成向量。

Example usage:
python generate_vector_refer_shuffle.py \
    --model_name qwen3vl \
    --model_size 8b \
    --layers $(seq 0 35) \
    --behavior_paths ../PAIRS/refer_shuffle_exp/LIUH0115_qwen3vl_8b_inst_it_image_refer_shuffle_n1024 \
    --output_dir refer_shuffle_exp \
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
    REFER_SHUFFLE,
)
from datasets import load_from_disk
from utils.mwrapper_map import (
    model_to_wrapper_map
)
from utils.conversation import (
    INST_IT_IMAGE_SYSTEM_PROMPT,
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


class ReferShuffleDataset(Dataset):
    """用于 Exact Matching vs Prompt Shuffle 对比的数据集类"""
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
            image_vpt_path = item["image_vpt"]  # 正确匹配的图像
            image_shuffle_path = item["image_shuffle"]  # 打乱匹配的图像
            instruction = item["instruction"]  # 指令文本（两个都用相同的指令）
            
            # 为VPT和Shuffle分别生成token
            # 两个都使用包含ID规则的 system prompt（因为都有ID框）
            # 两个都使用相同的 instruction（要求使用IDs）
            tokens_vpt = self.prompt_to_tokens(
                image_vpt_path, "image", instruction, 
                system_prompt=INST_IT_IMAGE_SYSTEM_PROMPT,  # 有ID规则
                verbose=False
            )
            tokens_shuffle = self.prompt_to_tokens(
                image_shuffle_path, "image", instruction,
                system_prompt=INST_IT_IMAGE_SYSTEM_PROMPT,  # 有ID规则（两个都用相同的）
                verbose=False
            )
            return image_vpt_path, image_shuffle_path, "image", tokens_vpt, tokens_shuffle, instruction
        elif self.data_name == "inst_it_video":
            video_vpt_path = item["video_vpt"]
            video_shuffle_path = item["video_shuffle"]
            instruction = item["instruction"]
            tokens_vpt = self.prompt_to_tokens(video_vpt_path, "video", instruction, system_prompt=INST_IT_VIDEO_SYSTEM_PROMPT, verbose=False)
            tokens_shuffle = self.prompt_to_tokens(video_shuffle_path, "video", instruction, system_prompt=INST_IT_VIDEO_SYSTEM_PROMPT, verbose=False)
            return video_vpt_path, video_shuffle_path, "video", tokens_vpt, tokens_shuffle, instruction
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
    """生成并保存 Exact Matching vs Prompt Shuffle 的 steering vectors"""
    if not os.path.exists(get_vector_dir(behavior, model_name, model_size, output_dir)):
        os.makedirs(get_vector_dir(behavior, model_name, model_size, output_dir))

    model.set_save_internal_decodings(False)
    model.reset_all()

    vpt_activations = dict([(layer, []) for layer in layers])  # 正确匹配图的激活
    shuffle_activations = dict([(layer, []) for layer in layers])  # 打乱匹配图的激活

    dataset = ReferShuffleDataset(
        model,
        behavior_path,
        model_name,
    )

    for vpt_path, shuffle_path, vision_type, tokens_vpt, tokens_shuffle, instruction in tqdm(dataset, desc="Processing image pairs"):
        tokens_vpt = tokens_vpt.to(model.device)
        tokens_shuffle = tokens_shuffle.to(model.device)
        
        # 处理正确匹配图（VPT）- 使用相同的指令和system prompt
        model.reset_all()
        model.get_logits(vpt_path, vision_type, tokens_vpt)
        for layer in layers:
            vpt_act = model.get_last_activations(layer)
            # 处理可能的3维激活 (batch, seq_len, hidden_dim) -> (seq_len, hidden_dim)
            if len(vpt_act.shape) == 3:
                vpt_act = vpt_act.squeeze(0)  # 移除batch维度
            assert len(vpt_act.shape) == 2, f"Expected 2D activations, got shape {vpt_act.shape}"
            # 取倒数第二个token的激活
            vpt_act = vpt_act[-2, :].detach().cpu()
            vpt_activations[layer].append(vpt_act)
        
        # 处理打乱匹配图（Shuffle）- 使用相同的指令和system prompt
        model.reset_all()
        model.get_logits(shuffle_path, vision_type, tokens_shuffle)
        for layer in layers:
            shuffle_act = model.get_last_activations(layer)
            # 处理可能的3维激活 (batch, seq_len, hidden_dim) -> (seq_len, hidden_dim)
            if len(shuffle_act.shape) == 3:
                shuffle_act = shuffle_act.squeeze(0)  # 移除batch维度
            assert len(shuffle_act.shape) == 2, f"Expected 2D activations, got shape {shuffle_act.shape}"
            shuffle_act = shuffle_act[-2, :].detach().cpu()
            shuffle_activations[layer].append(shuffle_act)
        
        if verbose:
            print(f"[VPT (Exact Match)]: {vpt_path}")
            print(f"[Shuffle (False Match)]: {shuffle_path}")
            print(f"[Instruction]: {instruction}")
            print("-" * 80)

    # 计算并保存向量
    for layer in layers:
        all_vpt_layer = t.stack(vpt_activations[layer])
        all_shuffle_layer = t.stack(shuffle_activations[layer])
        # Steering vector = mean(VPT激活 - Shuffle激活)
        # 这表示"正确匹配相对于打乱匹配"的激活方向
        vec = (all_vpt_layer - all_shuffle_layer).mean(dim=0)
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
    生成并保存 Exact Matching vs Prompt Shuffle 的 steering vectors
    
    Args:
        layers: 要生成向量的层列表
        wrapper: 模型包装器
        model_name: 模型名称
        model_size: 模型大小
        use_flash_attn: 是否使用flash attention
        behavior_paths: 行为数据路径列表（对于refer_shuffle，通常只有一个）
        output_dir: 输出目录
        verbose: 是否输出详细信息
    """
    model = wrapper(
        name=model_name,
        size=model_size,
        use_flash_attn=use_flash_attn
    )
    
    # 对于 refer_shuffle，我们使用固定的 behavior 名称
    behavior = REFER_SHUFFLE
    
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
