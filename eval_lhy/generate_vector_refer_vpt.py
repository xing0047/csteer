"""
Generate steering vectors for Refer VPT vs Raw.
The vector is computed from activation differences between annotated images (images_vpt) and the corresponding raw images (images_raw).

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
    """Dataset for Refer VPT vs Raw comparisons."""
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
        """Convert prompts to tokens (no model output; we only extract activations)."""
        tokens = self.wrapper.get_tokens_for_compare(
            image_or_video_path, 
            vision_type, 
            instruction, 
            model_output="",  # empty output: we only need activations
            system_prompt=system_prompt,  # allow custom system prompt
            verbose=verbose
        )
        return t.tensor(tokens).unsqueeze(0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        if self.data_name == "inst_it_image":
            image_vpt_path = item["image_vpt"]  # annotated image with numeric IDs
            image_raw_path = item["image_raw"]  # raw image
            instruction_vpt = item["instruction_vpt"]  # instruction for annotated image (requires IDs)
            instruction_raw = item["instruction_raw"]  # instruction for raw image (does not require IDs)
            
            # Tokenize VPT and Raw separately:
            # - VPT: system prompt includes ID rules
            # - Raw: system prompt without ID rules
            tokens_vpt = self.prompt_to_tokens(
                image_vpt_path, "image", instruction_vpt, 
                system_prompt=INST_IT_IMAGE_SYSTEM_PROMPT,  # with ID rules
                verbose=False
            )
            tokens_raw = self.prompt_to_tokens(
                image_raw_path, "image", instruction_raw,
                system_prompt=INST_IT_IMAGE_SYSTEM_PROMPT_NO_ID,  # without ID rules
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
    """Generate and save steering vectors for Refer VPT vs Raw."""
    if not os.path.exists(get_vector_dir(behavior, model_name, model_size, output_dir)):
        os.makedirs(get_vector_dir(behavior, model_name, model_size, output_dir))

    model.set_save_internal_decodings(False)
    model.reset_all()

    vpt_activations = dict([(layer, []) for layer in layers])  # activations for annotated (VPT) images
    raw_activations = dict([(layer, []) for layer in layers])  # activations for raw images

    dataset = ReferVPTDataset(
        model,
        behavior_path,
        model_name,
    )

    for vpt_path, raw_path, vision_type, tokens_vpt, tokens_raw, instruction_vpt, instruction_raw in tqdm(dataset, desc="Processing image pairs"):
        tokens_vpt = tokens_vpt.to(model.device)
        tokens_raw = tokens_raw.to(model.device)
        
        # Annotated (VPT) branch: instruction requires IDs
        model.reset_all()
        model.get_logits(vpt_path, vision_type, tokens_vpt)
        for layer in layers:
            vpt_act = model.get_last_activations(layer)
            # Handle possible 3D activations (batch, seq_len, hidden_dim) -> (seq_len, hidden_dim)
            if len(vpt_act.shape) == 3:
                vpt_act = vpt_act.squeeze(0)  # remove batch dimension
            assert len(vpt_act.shape) == 2, f"Expected 2D activations, got shape {vpt_act.shape}"
            # Use the second-to-last token (often right before answer generation).
            vpt_act = vpt_act[-2, :].detach().cpu()
            vpt_activations[layer].append(vpt_act)
        
        # Raw branch: instruction does not require IDs
        model.reset_all()
        model.get_logits(raw_path, vision_type, tokens_raw)
        for layer in layers:
            raw_act = model.get_last_activations(layer)
            # Handle possible 3D activations (batch, seq_len, hidden_dim) -> (seq_len, hidden_dim)
            if len(raw_act.shape) == 3:
                raw_act = raw_act.squeeze(0)  # remove batch dimension
            assert len(raw_act.shape) == 2, f"Expected 2D activations, got shape {raw_act.shape}"
            raw_act = raw_act[-2, :].detach().cpu()
            raw_activations[layer].append(raw_act)
        
        if verbose:
            print(f"[VPT Image]: {vpt_path}")
            print(f"[Raw Image]: {raw_path}")
            print(f"[Instruction VPT]: {instruction_vpt}")
            print(f"[Instruction Raw]: {instruction_raw}")
            print("-" * 80)

    # Compute and save vectors
    for layer in layers:
        all_vpt_layer = t.stack(vpt_activations[layer])
        all_raw_layer = t.stack(raw_activations[layer])
        # Steering vector = mean(VPT - Raw), i.e. the direction of "annotated vs raw".
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
    Generate and save steering vectors for Refer VPT vs Raw.
    
    Args:
        layers: Layers to generate vectors for
        wrapper: Model wrapper
        model_name: Model name
        model_size: Model size
        use_flash_attn: Whether to enable flash attention
        behavior_paths: Behavior dataset paths (for refer_vpt, usually just one)
        output_dir: Output directory
        verbose: Whether to print verbose logs
    """
    model = wrapper(
        name=model_name,
        size=model_size,
        use_flash_attn=use_flash_attn
    )
    
    # For refer_vpt we use a fixed behavior name
    behavior = REFER_VPT
    
    # Usually there is only one behavior_path
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

