import os
import re
import torch as t
import transformers
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
)
from utils.helpers import (
    find_instruction_end_postion, 
    get_model_path
)
from typing import Optional, List
from PIL import Image
from utils.prompts import prompt_template
from utils.conversation import (
    INST_IT_IMAGE_SYSTEM_PROMPT,
    INST_IT_VIDEO_SYSTEM_PROMPT,
)
from utils.overlay import overlay_visual_prompt
from utils.tokenize import find_marker_pos
from openai import OpenAI
import base64
from io import BytesIO
def pil_to_base64(img: Image.Image, fmt="PNG") -> str:
    buffer = BytesIO()
    img.save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
def resize_image(image, max_size=512):
    """
    Resize only when the image long edge exceeds max_size.
    """
    width, height = image.size
    
    # Resize only when long edge > max_size
    if max(width, height) > max_size:
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    return image

ADD_FROM_POS_CHAT="<|im_start|>assistant\n"

class OpenAIWrapper:
    def __init__(
        self,
        name: str = 'gpt-4o',
    ): 
        self.name = name
        # Steering code expects a `.device` attribute when it tries to move
        # steering vectors. OpenAI models are remote, so we keep vectors
        # on CPU and (optionally) no-op steering activations.
        self.device = t.device("cpu")
        # self.client = OpenAI(
        #     api_key="sk-yoWMcHhKxjSX31HVUVtVyR1uyac0Az4mE8SLU9ukNbfZDVgl",
        #     base_url="https://api.chatanywhere.tech/v1",
        # )
        self.client = OpenAI(
            api_key="sk-eMupyyt3u0hHMTea6RREjxSarex4Z7dUgUR15iUjNn9wl9Fu",
            base_url="https://api.chatanywhere.tech/v1",
        )


    def reset_all(self) -> None:
        # No internal model state to reset for remote OpenAI calls.
        return

    def set_add_activations(self, layer: int, value) -> None:
        # This repo's steering mechanism applies activation deltas to local
        # transformer models. For remote OpenAI calls we currently treat this
        # as a no-op (use `--noref` to skip vector loading entirely).
        return

    def _generate_with_chat_completions(self, content) -> str:
        completion = self.client.chat.completions.create(
            model=self.name,
            messages=[
                {
                    "role": "user",
                    "content": content,
                }
            ],
            stream=False,
        )
        choices = getattr(completion, "choices", None) or []
        if len(choices) == 0:
            raise RuntimeError("chat.completions: empty choices")
        msg = getattr(choices[0], "message", None)
        if msg is None:
            raise RuntimeError("chat.completions: missing message")
        text = getattr(msg, "content", None)
        if text is None or (isinstance(text, str) and not text.strip()):
            raise RuntimeError("chat.completions: empty message content")
        return text

    def _responses_text(self, out, use_last_block: bool = True) -> str:
        """Parse text from Responses API object; raise if missing."""
        out_list = getattr(out, "output", None)
        if not out_list:
            raise RuntimeError("responses: empty output")
        block = out_list[-1] if use_last_block else out_list[0]
        parts = getattr(block, "content", None)
        if not parts:
            raise RuntimeError("responses: empty content")
        first = parts[0]
        text = getattr(first, "text", None)
        if text is None or (isinstance(text, str) and not str(text).strip()):
            raise RuntimeError("responses: empty text")
        return text

    def _sample_video_frames(self, frame_paths: List[str], max_frames: int = 8) -> List[Image.Image]:
        """Load uniformly sampled frames from a video frame-path list."""
        if len(frame_paths) == 0:
            raise ValueError("Empty frame list for video input.")
        if len(frame_paths) <= max_frames:
            sampled_paths = frame_paths
        else:
            sampled_paths = []
            n = len(frame_paths)
            for i in range(max_frames):
                idx = int(round(i * (n - 1) / (max_frames - 1)))
                sampled_paths.append(frame_paths[idx])
        frames = [resize_image(Image.open(path).convert("RGB")) for path in sampled_paths]
        return frames
    
    def generate_text(self, image_or_video_path, vision_type, question, in_query: bool=False,
            marker_only: bool=False, not_marker: bool=False, verbose: bool=False, system_prompt: Optional[str]=None, **kwargs) -> str:
        if vision_type == "image":
            assert isinstance(image_or_video_path, Image.Image)
            image_or_video_path = resize_image(image_or_video_path)
            b64_image = pil_to_base64(image_or_video_path)
            if self.name == "gemini-2.5-pro":
                generated = self._generate_with_chat_completions(
                    [
                        {"type": "text", "text": question},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64_image}"},
                        },
                    ]
                )
            else:
                out = self.client.responses.create(
                    model=self.name,
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_image", "image_url": f"data:image/png;base64,{b64_image}"},
                                {"type": "input_text", "text": question},
                            ],
                        }
                    ],
                )
                if self.name == "gpt-4o":
                    generated = self._responses_text(out, use_last_block=False)
                elif self.name == "o3":
                    generated = self._responses_text(out, use_last_block=True)
                else:
                    raise NotImplementedError
            if verbose:
                print(f"[q]: \033[31m{question}\033[0m")
        elif vision_type == "multi_image":
            assert isinstance(image_or_video_path, list)
            assert isinstance(image_or_video_path[0], Image.Image)
            image_or_video_path = [resize_image(image) for image in image_or_video_path]
            b64_images = [pil_to_base64(image) for image in image_or_video_path]
            if self.name == "gemini-2.5-pro":
                content = [{"type": "text", "text": question}]
                content.extend(
                    [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64_image}"},
                        }
                        for b64_image in b64_images
                    ]
                )
                generated = self._generate_with_chat_completions(content)
            else:
                content = [
                    {"type": "input_image", "image_url": f"data:image/png;base64,{b64_image}"} for b64_image in b64_images
                ]
                content.extend(
                    [
                        {"type": "input_text", "text": question},
                    ]
                )
                out = self.client.responses.create(
                    model=self.name,
                    input=[
                        {
                            "role": "user",
                            "content": content
                        }
                    ],
                )
                if self.name == "gpt-4o":
                    generated = self._responses_text(out, use_last_block=False)
                elif self.name == "o3":
                    generated = self._responses_text(out, use_last_block=True)
                else:
                    raise NotImplementedError
            if verbose:
                print(f"[q]: \033[31m{question}\033[0m")
        elif vision_type == "video":
            assert isinstance(image_or_video_path, list)
            assert isinstance(image_or_video_path[0], str)
            sampled_frames = self._sample_video_frames(image_or_video_path, max_frames=8)
            b64_images = [pil_to_base64(image) for image in sampled_frames]
            if self.name == "gemini-2.5-pro":
                content = [{"type": "text", "text": question}]
                content.extend(
                    [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64_image}"},
                        }
                        for b64_image in b64_images
                    ]
                )
                generated = self._generate_with_chat_completions(content)
            else:
                content = [
                    {"type": "input_image", "image_url": f"data:image/png;base64,{b64_image}"} for b64_image in b64_images
                ]
                content.extend(
                    [
                        {"type": "input_text", "text": question},
                    ]
                )
                out = self.client.responses.create(
                    model=self.name,
                    input=[
                        {
                            "role": "user",
                            "content": content
                        }
                    ],
                )
                if self.name == "gpt-4o":
                    generated = self._responses_text(out, use_last_block=False)
                elif self.name == "o3":
                    generated = self._responses_text(out, use_last_block=True)
                else:
                    raise NotImplementedError
            if verbose:
                print(f"[q]: \033[31m{question}\033[0m")
        else:
            raise NotImplementedError
        return generated

    def get_inputs(self, type, dataloader_item):
        if type == 'inst_it_image':
            assert 'image_path' in dataloader_item
            image_file = os.path.join("../DATA/Inst-It-Dataset", dataloader_item['image_path'])
            image = Image.open(image_file).convert('RGB')
            prompt = None
        elif type == 'blink_image_mc_qa':
            assert 'pil_images' in dataloader_item
            images = dataloader_item['pil_images']

            pre = prompt_template["blink_image_mc"]["pre_prompt"]
            post = prompt_template["image_mc"]["post_prompt"]
            options = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(dataloader_item['choices'])])
            prompt = f"{pre}\n{dataloader_item['question']}\n{options}\n{post}"
            return images, prompt
        elif type == 'cvbench_image_mc_qa':
            assert 'pil_image' in dataloader_item
            image = dataloader_item['pil_image']

            pre = prompt_template["cvbench_image_mc"]["pre_prompt"]
            post = prompt_template["image_mc"]["post_prompt"]
            prompt = f"{pre}\n{dataloader_item['question']}\n{post}"
            return image, prompt
        elif type == 'inst_it_image_mc_qa':
            assert 'pil_image' in dataloader_item
            image = dataloader_item['pil_image'].convert('RGB')

            pre  = prompt_template["image_mc"]["pre_prompt"]
            post = prompt_template["image_mc"]["post_prompt"]
            options = (
                f"A. {dataloader_item['choice_a']}\n"
                f"B. {dataloader_item['choice_b']}\n"
                f"C. {dataloader_item['choice_c']}\n"
                f"D. {dataloader_item['choice_d']}"
            )
            prompt = f"{pre}\n{dataloader_item['question']}\n{options}\n{post}"
            return image, prompt
        elif type == 'inst_it_image_oe_qa':
            assert 'pil_image' in dataloader_item
            image = dataloader_item['pil_image'].convert('RGB')

            pre  = prompt_template["image_oe"]["pre_prompt"]
            post = prompt_template["image_oe"]["post_prompt"]
            
            prompt = f"{pre}\n{post}\n{dataloader_item['question']}"
            return image, prompt
        elif type == 'inst_it_video_mc_qa':
            video_root = f"../DATA/Inst-It-Bench/{dataloader_item['video_path']}"
            frames = os.listdir(video_root)
            frames = sorted(frames)
            video_path = [os.path.join(video_root, frame) for frame in frames]
            
            option_str = ""
            for option, ans_str in dataloader_item["options"].items():
                option_str += f"{option}. {ans_str}\n"
            
            pre = prompt_template["video_mc"]["pre_prompt"]
            post = prompt_template["video_mc"]["post_prompt"]

            prompt = f"{pre}\n{dataloader_item['question']}\n{option_str}{post}"
            prompt = re.sub(r'<(\d+)>', r'<\1.0 seconds>', prompt)
            return video_path, prompt
        elif type == 'inst_it_video_oe_qa':
            video_root = f"../DATA/Inst-It-Bench/{dataloader_item['video_path']}"
            frames = os.listdir(video_root)
            frames = sorted(frames)
            video_path = [os.path.join(video_root, frame) for frame in frames]
            
            pre = prompt_template["video_oe"]["pre_prompt"]
            post = prompt_template["video_oe"]["post_prompt"]
            
            prompt = f"{pre}\n{post}\n{dataloader_item['question']}"
            prompt = re.sub(r'<(\d+)>', r'<\1.0 seconds>', prompt)
            return video_path, prompt
        elif type == 'gar_image_mc_qa':
            assert 'pil_image' in dataloader_item
            image = dataloader_item['pil_image'].convert('RGB')
            pre  = prompt_template["image_mc"]["pre_prompt"]
            post = prompt_template["image_mc"]["post_prompt"]
            options = "\n".join(dataloader_item['choices'])
            prompt = f"{pre}\n{dataloader_item['question']}\n{options}\n{post}"
            # post-process: from <Prompt*> to [*]
            prompt = re.sub(r'<Prompt(\d+)>', r'[\1]', prompt)
            # post-process: overlay visual prompts
            image = overlay_visual_prompt(image, dataloader_item["mask_rles"], format="boxnum")
            return image, prompt
        elif type == 'gar_image_simple_oe_qa':
            image_file = os.path.join(dataloader_item['image_path'])
            image = Image.open(image_file).convert('RGB')
            pre = prompt_template["image_oe"]["pre_prompt"]
            post = prompt_template["image_oe"]["post_prompt"]
            prompt = f"{pre}\n{post}\n{dataloader_item['question']}"
            prompt = re.sub(r'<Prompt(\d+)>', r'[\1]', prompt)
            image = overlay_visual_prompt(image, dataloader_item["mask_rles"], format="boxnum")
            return image, prompt
        elif type == 'gar_image_detail_oe_qa':
            image_file = os.path.join(dataloader_item['image_path'])
            image = Image.open(image_file).convert('RGB')
            pre = prompt_template["image_oe"]["pre_prompt"]
            post = prompt_template["image_oe"]["post_prompt"]
            question = "Describe [0] in detail, including the relationship with [1]."
            prompt = f"{pre}\n{post}\n{question}"
            return image, prompt
        elif type == 'vip_image_oe_qa':
            image_file = os.path.join(dataloader_item['image_path'])
            image = Image.open(image_file).convert('RGB')
            pre = prompt_template["vip_image_oe"]["pre_prompt"]
            post = prompt_template["vip_image_oe"]["post_prompt"]
            question = dataloader_item["question"]
            prompt = f"{pre}\n{post}\n{question}"
            return image, prompt
        else:
            raise ValueError(f'{type} not supported.')
