"""
Generates steering vectors for each layer of the model by averaging the activations of all the positive and negative examples.

Example usage:
python generate_vector.py --layers $(seq 0 31) --model_name internvl3_5 --behaviors refer --behavior_paths ... --output_dir ...
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
)
from datasets import load_from_disk
from utils.mwrapper_map import (
    model_to_wrapper_map
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, choices=["internvl3_5", "qwen3vl"]
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
        "--behaviors", nargs="+", type=str, default="refer"
    )
    parser.add_argument(
        "--behavior_paths", nargs="+", type=str, required=True
    )
    parser.add_argument(
        "--output_dir", type=str, required=True
    )
    return parser.parse_args()


class ComparisonDataset(Dataset):
    def __init__(self, wrapper, data_path, model_name):
        self.data = load_from_disk(data_path)
        self.wrapper = wrapper
        self.model = wrapper.model
        if "refcocog" in data_path:
            self.data_name = "refcocog"
            self.vision_type = "image"
        elif "inst_it_image" in data_path:
            self.data_name = "inst_it_image"
            self.vision_type = "image"
        elif "inst_it_video" in data_path:
            self.data_name = "inst_it_video"
            self.vision_type = "video"
        else:
            raise ValueError
        self.model_name = model_name
        self.tokenizer = wrapper.tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def prompt_to_tokens(self, image_or_video_path, vision_type, instruction, model_output, verbose=False):
        tokens = self.wrapper.get_tokens_for_compare(image_or_video_path, vision_type, instruction, model_output, verbose=verbose)
        return t.tensor(tokens).unsqueeze(0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        p_text = item["answer_matching_behavior"]
        n_text = item["answer_not_matching_behavior"]
        if self.data_name == "inst_it_image":
            q_text = f"Which description matches the image better?\n(A) {item['caption_gt']}\n(B) {item['caption_generated']}"
            image_path = item["image_path"] = item["image"]
            p_tokens = self.prompt_to_tokens(image_path, "image", q_text, p_text, verbose=True)
            n_tokens = self.prompt_to_tokens(image_path, "image", q_text, n_text, verbose=True)
            return image_path, "image", p_tokens, n_tokens
        elif self.data_name == "inst_it_video":
            q_text = f"Which description matches the video better?\n(A) {item['caption_gt']}\n(B) {item['caption_generated']}"
            video_path = item["video"]
            p_tokens = self.prompt_to_tokens(video_path, "video", q_text, p_text, verbose=True)
            n_tokens = self.prompt_to_tokens(video_path, "video", q_text, n_text, verbose=True)
            return video_path, "video", p_tokens, n_tokens
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
):
    if not os.path.exists(get_vector_dir(behavior, model_name, model_size, output_dir)):
        os.makedirs(get_vector_dir(behavior, model_name, model_size, output_dir))

    model.set_save_internal_decodings(False)
    model.reset_all()

    pos_activations = dict([(layer, []) for layer in layers])
    neg_activations = dict([(layer, []) for layer in layers])

    dataset = ComparisonDataset(
        model,
        behavior_path,
        model_name,
    )

    for image_or_video_path, vision_type, p_tokens, n_tokens in tqdm(dataset, desc="Processing prompts"):
        p_tokens = p_tokens.to(model.device)
        n_tokens = n_tokens.to(model.device)
        model.reset_all()
        model.get_logits(image_or_video_path, vision_type, p_tokens)
        for layer in layers:
            p_activations = model.get_last_activations(layer).squeeze(0)
            assert len(p_activations.shape) == 2
            p_activations = p_activations[-2, :].detach().cpu()
            pos_activations[layer].append(p_activations)
        model.reset_all()
        model.get_logits(image_or_video_path, vision_type, n_tokens)
        for layer in layers:
            n_activations = model.get_last_activations(layer).squeeze(0)
            assert len(n_activations.shape) == 2
            n_activations = n_activations[-2, :].detach().cpu()
            neg_activations[layer].append(n_activations)

    for layer in layers:
        all_pos_layer = t.stack(pos_activations[layer])
        all_neg_layer = t.stack(neg_activations[layer])
        vec = (all_pos_layer - all_neg_layer).mean(dim=0)
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
    behaviors: List[str],
    behavior_paths: List[str],
    output_dir: str,
):
    """
    layers: list of layers to generate vectors for
    save_activations: if True, save the activations for each layer
    wrapper: wrapper of the model to use
    behaviors: behaviors to generate vectors for
    """
    model = wrapper(
        name=model_name,
        size=model_size,
        use_flash_attn=use_flash_attn
    )
    for behavior, behavior_path in zip(behaviors, behavior_paths):
        generate_save_vectors_for_behavior(
            layers, model_name, model_size, behavior, 
            behavior_path, model, output_dir
        )


if __name__ == "__main__":
    
    args = parse_args()
    wrapper = model_to_wrapper_map[args.model_name]["8b"]
    generate_save_vectors(
        args.layers,
        wrapper,
        args.model_name,
        "8b",
        args.use_flash_attn,
        args.behaviors,
        args.behavior_paths,
        args.output_dir
    )
