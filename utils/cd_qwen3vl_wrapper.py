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
    add_vector_from_position, 
    find_instruction_end_postion, 
    get_model_path
)
from typing import Optional
from PIL import Image
from utils.prompts import prompt_template
from utils.conversation import (
    INST_IT_IMAGE_SYSTEM_PROMPT,
    INST_IT_VIDEO_SYSTEM_PROMPT,
)
from utils.overlay import overlay_visual_prompt
from utils.tokenize import find_marker_pos

ADD_FROM_POS_CHAT="<|im_start|>assistant\n"


class AttnWrapper(t.nn.Module):
    """
    Wrapper for attention mechanism to save activations
    """
    def __init__(self, attn):
        super().__init__()
        self.attn = attn
        self.activations = None

    def forward(self, *args, **kwargs):
        output = self.attn(*args, **kwargs)
        self.activations = output[0]
        return output


class BlockOutputWrapper(t.nn.Module):
    """
    Wrapper for block to save activations and unembed them
    """
    def __init__(self, block, unembed_matrix, norm, tokenizer):
        super().__init__()
        self.block = block
        self.unembed_matrix = unembed_matrix
        self.norm = norm
        self.tokenizer = tokenizer
        self.block.self_attn = AttnWrapper(self.block.self_attn)
        self.mlp = self.block.mlp 
        self.input_layernorm = self.block.input_layernorm
        self.post_attention_layernorm = self.block.post_attention_layernorm
        self.attention_type = self.block.self_attn.attn.config._attn_implementation

        self.attn_out_unembedded = None
        self.intermediate_resid_unembedded = None
        self.mlp_out_unembedded = None
        self.block_out_unembedded = None

        self.activations = None
        self.add_activations = None
        self.from_position = None

        self.save_internal_decodings = False

        self.calc_dot_product_with = None
        self.dot_products = []

        self.in_query_mask = None
        self.in_decode_steer_flag = False

    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)
        self.activations = output
        if self.calc_dot_product_with is not None:
            last_token_activations = self.activations[0, -1, :]
            decoded_activations = self.unembed_matrix(self.norm(last_token_activations))
            top_token_id = t.topk(decoded_activations, 1)[1][0]
            top_token = self.tokenizer.decode(top_token_id)
            dot_product = t.dot(last_token_activations, self.calc_dot_product_with) / (
                t.norm(last_token_activations) * t.norm(self.calc_dot_product_with)
            )
            self.dot_products.append((top_token, dot_product.cpu().item()))
        if self.add_activations is not None:
            augmented_output = add_vector_from_position(
                matrix=output,
                vector=self.add_activations,
                position_ids=kwargs["position_ids"],
                from_pos=self.from_position,
                is_qwen_vl=True,
                in_query_mask=self.in_query_mask,
                in_decode_steer_flag=self.in_decode_steer_flag,
            )
            output = augmented_output
        
        if not self.save_internal_decodings:
            return output

        # Whole block unembedded
        self.block_output_unembedded = self.unembed_matrix(self.norm(output[0]))

        # Self-attention unembedded
        attn_output = self.block.self_attn.activations
        self.attn_out_unembedded = self.unembed_matrix(self.norm(attn_output))

        # Intermediate residual unembedded
        attn_output += args[0]
        self.intermediate_resid_unembedded = self.unembed_matrix(self.norm(attn_output))

        # MLP unembedded
        mlp_output = self.block.mlp(self.post_attention_layernorm(attn_output))
        self.mlp_out_unembedded = self.unembed_matrix(self.norm(mlp_output))

        return output

    def add(self, activations):
        self.add_activations = activations

    def reset(self):
        self.add_activations = None
        self.activations = None
        self.block.self_attn.activations = None
        self.from_position = None
        self.calc_dot_product_with = None
        self.dot_products = []
        self.in_query_mask = None
        self.in_decode_steer_flag = False


class Qwen3VL_Wrapper:
    def __init__(
        self,
        name: str = 'qwen3vl',
        size: str = "8b",
        use_flash_attn: bool = True,
    ):
        self.device = "cuda" if t.cuda.is_available() else "cpu"
        self.model_name_path = get_model_path(name, size)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_path, 
            use_fast=False
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_name_path
        )
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_name_path, 
            torch_dtype="auto", 
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2" if use_flash_attn else "eager", 
            device_map="auto"
        ).to(self.device)
        self.END_STR = t.tensor(
            self.tokenizer.encode(ADD_FROM_POS_CHAT)[1:]
        ).to(self.device)
        self.eos_token_id = self.model.config.text_config.eos_token_id
        for i, layer in enumerate(self.model.model.language_model.layers):
            self.model.model.language_model.layers[i] = BlockOutputWrapper(
                layer, self.model.lm_head, self.model.model.language_model.norm, self.tokenizer
            )

    def set_save_internal_decodings(self, value: bool):
        for layer in self.model.model.language_model.layers:
            layer.save_internal_decodings = value

    def set_from_positions(self, pos: int):
        for layer in self.model.model.language_model.layers:
            layer.from_position = pos

    def set_in_query_mask(self, example_mask, marker_pos):
        in_query_mask = t.zeros_like(example_mask)
        for _, pos in marker_pos:
            in_query_mask[:, pos] = True
        for layer in self.model.model.language_model.layers:
            layer.in_query_mask = in_query_mask

    def generate(
        self,
        inputs,
        tokens,
        **kwargs,
    ):
        with t.no_grad():
            instr_pos = find_instruction_end_postion(tokens[0], self.END_STR)
            self.set_from_positions(instr_pos)
            kwargs.update(
                {'eos_token_id': self.eos_token_id}
            )
            generated = self.model.generate(**inputs, **kwargs)
            generated_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated)
            ]
            generated_str = self.tokenizer.batch_decode(generated_trimmed, skip_special_tokens=True)[0]
            return generated_str

    def generate_text(
        self,
        image_or_video_path,
        vision_type,
        question,
        in_query_steer: bool = False,
        marker_only: bool = False,
        not_marker: bool = False,
        verbose: bool = False,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> str:
        if vision_type == "image":
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_or_video_path,
                        },
                        {"type": "text", "text": question},
                    ],
                }
            ]
            if verbose:
                print(f"[q]: \033[31m{question}\033[0m")
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
        elif vision_type == "multi_image":
            content = [
                {"type": "image", "image": image} for image in image_or_video_path
            ]
            content.extend(
                [
                    {"type": "text", "text": question},
                ]
            )
            messages = [
                {
                    "role": "user",
                    "content": content
                }
            ]
            if verbose:
                print(f"[q]: \033[31m{question}\033[0m")
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
        elif vision_type == "video":
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": image_or_video_path,
                        },
                        {"type": "text", "text": question},
                    ],
                }
            ]
            if verbose:
                print(f"[q]: \033[31m{question}\033[0m")
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                do_sample_frames=False,
            )
        else:
            raise NotImplementedError
        inputs = inputs.to(self.model.device)
        tokens = t.tensor(inputs.input_ids).cuda()
        if in_query_steer:
            marker_pos = find_marker_pos(tokens, self.tokenizer)
            self.set_in_query_mask(inputs.attention_mask, marker_pos)
        assert len(tokens.shape) == 2
        if marker_only and not_marker:
            raise ValueError("marker_only and not_marker cannot both be True")
        if marker_only:
            from utils.msteer_sample import msteer_sample
            transformers.generation.utils.GenerationMixin._sample = msteer_sample
            self.model._msteer_marker_only = True
            self.model._msteer_not_marker = False
            kwargs.update({"tokenizer": self.tokenizer})
        elif not_marker:
            from utils.msteer_sample import msteer_sample
            transformers.generation.utils.GenerationMixin._sample = msteer_sample
            for layer in self.model.model.language_model.layers:
                layer.in_decode_steer_flag = True
            self.model._msteer_marker_only = False
            self.model._msteer_not_marker = True
            kwargs.update({"tokenizer": self.tokenizer})
        else:
            self.model._msteer_marker_only = False
            self.model._msteer_not_marker = False
            for layer in self.model.model.language_model.layers:
                layer.in_decode_steer_flag = True
        return self.generate(
            inputs=inputs,
            tokens=tokens,
            **kwargs,
        )

    def get_logits(self, image_or_video_path, vision_type, tokens):
        with t.no_grad():
            instr_pos = find_instruction_end_postion(tokens[0], self.END_STR)
            self.set_from_positions(instr_pos)
            attention_mask = t.ones_like(tokens)
            if vision_type == "image":
                pixel_values, image_grid_thw = self.get_pixel_values(image_or_video_path)
                logits = self.model.model(
                    input_ids=tokens,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                ).last_hidden_state
            elif vision_type == "video":
                pixel_values_videos, video_grid_thw = self.get_pixel_values_videos(image_or_video_path)
                logits = self.model.model(
                    input_ids=tokens,
                    attention_mask=attention_mask,
                    pixel_values_videos=pixel_values_videos,
                    video_grid_thw=video_grid_thw,
                ).last_hidden_state
            else:
                raise NotImplementedError
            return logits

    def get_inputs(self, type, dataloader_item):
        if type == 'inst_it_image':
            assert 'image_path' in dataloader_item
            image_file = os.path.join("../DATA/Inst-It-Dataset", dataloader_item['image_path'])
            image = Image.open(image_file).convert('RGB')
            prompt = None
        elif type == 'vip_bench_qa':
            raise NotImplementedError
            image_file = f"datasets/test/vipbench/bbox/images/{item['image']}"
            image = Image.open(image_file).convert('RGB')
            prompt = item['text']
        elif type == 'blink_image_mc_qa':
            assert 'pil_images' in dataloader_item
            images = dataloader_item['pil_images']

            pre = prompt_template["blink_image_mc"]["pre_prompt"]
            post = prompt_template["image_mc"]["post_prompt"]
            options = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(dataloader_item['choices'])])
            prompt = f"{pre}\n{dataloader_item['question']}\n{options}\n{post}"
            return images, prompt
        elif type == 'blink_image_think_mc_qa':
            assert 'pil_images' in dataloader_item
            images = dataloader_item['pil_images']

            pre = prompt_template["blink_image_think_mc"]["pre_prompt"]
            post = prompt_template["blink_image_think_mc"]["post_prompt"]
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
        elif type == 'cvbench_image_think_mc_qa':
            assert 'pil_image' in dataloader_item
            image = dataloader_item['pil_image']

            pre = prompt_template["cvbench_image_think_mc"]["pre_prompt"]
            post = prompt_template["cvbench_image_think_mc"]["post_prompt"]
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
        elif type == 'inst_it_image_think_mc_qa':
            assert 'pil_image' in dataloader_item
            image = dataloader_item['pil_image'].convert('RGB')

            pre  = prompt_template["image_think_mc"]["pre_prompt"]
            post = prompt_template["image_think_mc"]["post_prompt"]
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
            pre  = prompt_template["gar_mc"]["pre_prompt"]
            post = prompt_template["gar_mc"]["post_prompt"]
            options = "\n".join(dataloader_item['choices'])
            prompt = f"{pre}\n{dataloader_item['question']}\n{options}\n{post}"
            # post-process: from <Prompt*> to [*]
            prompt = re.sub(r'<Prompt(\d+)>', r'[\1]', prompt)
            # post-process: overlay visual prompts
            image = overlay_visual_prompt(image, dataloader_item["mask_rles"], format="boxnum")
            return image, prompt
        elif type == 'gar_image_think_mc_qa':
            assert 'pil_image' in dataloader_item
            image = dataloader_item['pil_image'].convert('RGB')

            pre  = prompt_template["image_think_mc"]["pre_prompt"]
            post = prompt_template["image_think_mc"]["post_prompt"]
            options = "\n".join(dataloader_item['choices'])
            prompt = f"{pre}\n{dataloader_item['question']}\n{options}\n{post}"
            # post-process for gar question from <Prompt*> to [*]
            prompt = re.sub(r'<Prompt(\d+)>', r'[\1]', prompt)
            # post-process for gar image for overlay visual prompts
            image = overlay_visual_prompt(image, dataloader_item["mask_rles"], format="boxnum")
            return image, prompt
        elif type == 'gar_image_simple_oe_qa':
            image_file = os.path.join(dataloader_item['image_path'])
            image = Image.open(image_file).convert('RGB')
            pre = prompt_template["gar_simple"]["pre_prompt"]
            post = prompt_template["gar_simple"]["post_prompt"]
            prompt = f"{pre}\n{post}\n{dataloader_item['question']}"
            prompt = re.sub(r'<Prompt(\d+)>', r'[\1]', prompt)
            image = overlay_visual_prompt(image, dataloader_item["mask_rles"], format="boxnum")
            return image, prompt
        elif type == 'gar_image_detail_oe_qa':
            image_file = os.path.join(dataloader_item['image_path'])
            image = Image.open(image_file).convert('RGB')
            pre = prompt_template["gar_detailed"]["pre_prompt"]
            post = prompt_template["gar_detailed"]["post_prompt"]
            question = "Describe [0] in detail, including the relationship with [1]."
            prompt = f"{pre}\n{post}\n{question}"
            return image, prompt
        else:
            raise ValueError(f'{type} not supported.')
    
    def get_tokens_for_compare(self, image_or_video_path, vision_type, instruction, model_output, verbose=False):
        if vision_type == "image":
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": INST_IT_IMAGE_SYSTEM_PROMPT},
                        {
                            "type": "image",
                            "image": image_or_video_path,
                        },
                        {"type": "text", "text": instruction},
                    ],
                }
            ]
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
        elif vision_type == "video":
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": INST_IT_VIDEO_SYSTEM_PROMPT},
                        {
                            "type": "video",
                            "video": image_or_video_path,
                        },
                        {"type": "text", "text": instruction},
                    ],
                }
            ]
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                do_sample_frames=False,
            )
        else:
            raise NotImplementedError

        model_output_ids = self.tokenizer.encode(model_output)
        return t.tensor(inputs.input_ids.squeeze().tolist() + model_output_ids)
    
    def get_pixel_values(self, image_path_or_image):
        if isinstance(image_path_or_image, str):
            image = Image.open(image_path_or_image).convert('RGB')
        elif isinstance(image_path_or_image, Image):
            image = image_path_or_image
        else:
            raise TypeError
        image_inputs = self.processor.image_processor(images=image, return_tensors="pt")
        pixel_values = image_inputs['pixel_values'].to(self.model.device)
        image_grid_thw = image_inputs['image_grid_thw'].to(self.model.device)
        return pixel_values, image_grid_thw

    def get_pixel_values_videos(self, video_path):
        assert isinstance(video_path, list)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                    },
                    {"type": "text", "text": "Describe this video."},
                ],
            }
        ]
        video_inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            do_sample_frames=False,
        )
        pixel_values_videos = video_inputs['pixel_values_videos'].to(self.model.device)
        video_grid_thw = video_inputs['video_grid_thw'].to(self.model.device)
        return pixel_values_videos, video_grid_thw

    def get_last_activations(self, layer):
        return self.model.model.language_model.layers[layer].activations
    
    def set_add_activations(self, layer, activations):
        self.model.model.language_model.layers[layer].add(activations)

    def reset_all(self):
        self.model._msteer_marker_only = False
        self.model._msteer_not_marker = False
        for layer in self.model.model.language_model.layers:
            layer.reset()
