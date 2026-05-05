import os
import re
import math
import torch as t
import transformers
import torchvision.transforms as T
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    AutoConfig,
)
from torchvision.transforms.functional import InterpolationMode
from utils.helpers import (
    add_vector_from_position, 
    find_instruction_end_postion, 
    get_model_path
)
from typing import Optional
from PIL import Image
from utils.prompts import prompt_template
from utils.conversation import (
    get_conv_template,
    INST_IT_IMAGE_SYSTEM_PROMPT,
    INST_IT_VIDEO_SYSTEM_PROMPT,
)
from utils.load_image import (
    internvl_load_frames
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
        self.block.self_attn  = AttnWrapper(self.block.self_attn)
        self.mlp = self.block.mlp 
        self.input_layernorm = self.block.input_layernorm
        self.post_attention_layernorm = self.block.post_attention_layernorm
        self.attention_type = self.block.attention_type

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


class InternVL3_5_Wrapper:
    def __init__(
        self,
        name: str = 'internvl3_5',
        size: str = "8b",
        use_flash_attn: bool = False,
        override_model_weights_path: Optional[str] = None,
    ):
        self.device = "cuda" if t.cuda.is_available() else "cpu"
        self.model_name_path = get_model_path(name, size)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_path, trust_remote_code=True, use_fast=False
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_path, torch_dtype=t.bfloat16, low_cpu_mem_usage=True,
            use_flash_attn=use_flash_attn, trust_remote_code=True
        )
        if override_model_weights_path is not None:
            self.model.load_state_dict(t.load(override_model_weights_path))
        self.model = self.model.to(self.device)
        self.END_STR = t.tensor(self.tokenizer.encode(ADD_FROM_POS_CHAT)[1:]).to(
            self.device
        )
        self.template = get_conv_template(self.model.template)
        self.eos_token_id = self.tokenizer.convert_tokens_to_ids(self.template.sep.strip())
        for i, layer in enumerate(self.model.language_model.model.layers):
            self.model.language_model.model.layers[i] = BlockOutputWrapper(
                layer, self.model.language_model.lm_head, self.model.language_model.model.norm, self.tokenizer
            )
    
    def split_model(self):
        device_map = {}
        world_size = t.cuda.device_count()
        config = AutoConfig.from_pretrained(self.model_name_path, trust_remote_code=True)
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

    def set_save_internal_decodings(self, value: bool):
        for layer in self.model.language_model.model.layers:
            layer.save_internal_decodings = value

    def set_from_positions(self, pos: int):
        for layer in self.model.language_model.model.layers:
            layer.from_position = pos
    
    def set_in_query_mask(self, example_mask, marker_pos):
        in_query_mask = t.zeros_like(example_mask)
        for _, pos in marker_pos:
            in_query_mask[:, pos] = True
        for layer in self.model.language_model.model.layers:
            layer.in_query_mask = in_query_mask
    
    def generate(self, pixel_values, tokens, attention_mask, **kwargs):
        with t.no_grad():
            instr_pos = find_instruction_end_postion(tokens[0], self.END_STR)
            self.set_from_positions(instr_pos)
            generated = self.model.generate(
                pixel_values=pixel_values, input_ids=tokens, attention_mask=attention_mask, **kwargs
            )
            generated_str = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
            generated_str = generated_str.split(self.template.sep.strip())[0].strip()
            return generated_str

    def generate_text(self, image_or_video_path, vision_type, question, in_query: bool=False,
            marker_only: bool=False, not_marker: bool=False, verbose: bool=False, system_prompt: Optional[str]=None, **kwargs) -> str:
        if vision_type == "image":
            pixel_values = self.get_pixel_values(image_or_video_path)
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
            question = f"<image>\n{question}"
            if verbose:
                print(f"[q]: \033[31m{question}\033[0m")
            generation_config = {
                'max_new_tokens': 1024, 'do_sample': False
            }
            img_context_token_id = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
            self.model.img_context_token_id = img_context_token_id

            template = get_conv_template(self.model.template)
            template.system_message = self.model.system_message
            eos_token_id = self.tokenizer.convert_tokens_to_ids(template.sep.strip())

            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            for num_patches in num_patches_list:
                image_tokens = "<img>" + "<IMG_CONTEXT>" * self.model.num_image_token * num_patches + "</img>"
                query = query.replace("<image>", image_tokens, 1)
            
            model_inputs = self.tokenizer(query, return_tensors="pt")
            input_ids = model_inputs["input_ids"].to(self.device)
            attention_mask = model_inputs["attention_mask"].to(self.device)
            generation_config["eos_token_id"] = eos_token_id
        elif vision_type == "multi_image":
            pixel_values_list = [self.get_pixel_values(pil) for pil in image_or_video_path]
            pixel_values = t.cat(pixel_values_list, dim=0)
            num_patches_list = [pixel_value.shape[0] for pixel_value in pixel_values_list]
            question_prefix = ""
            for pixel_value_id in range(len(num_patches_list)):
                question_prefix += f"Image-{str(pixel_value_id+1)}: <image>\n"
            question = question_prefix + question
            if verbose:
                print(f"[q]: \033[31m{question}\033[0m")
            generation_config = {
                'max_new_tokens': 1024, 'do_sample': False
            }
            img_context_token_id = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
            self.model.img_context_token_id = img_context_token_id

            template = get_conv_template(self.model.template)
            template.system_message = self.model.system_message
            eos_token_id = self.tokenizer.convert_tokens_to_ids(template.sep.strip())

            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            for num_patches in num_patches_list:
                image_tokens = "<img>" + "<IMG_CONTEXT>" * self.model.num_image_token * num_patches + "</img>"
                query = query.replace("<image>", image_tokens, 1)
            
            model_inputs = self.tokenizer(query, return_tensors="pt")
            input_ids = model_inputs["input_ids"].to(self.device)
            attention_mask = model_inputs["attention_mask"].to(self.device)
            generation_config["eos_token_id"] = eos_token_id
        elif vision_type == "video":
            pixel_values, num_patches_list = self.get_pixel_values_videos(image_or_video_path)
            question = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))]) + f"{question}"
            if verbose:
                print(f"[q]: \033[31m{question}\033[0m")
            generation_config = {
                'max_new_tokens': 1024, 'do_sample': False
            }
            img_context_token_id = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
            self.model.img_context_token_id = img_context_token_id

            template = get_conv_template(self.model.template)
            template.system_message = self.model.system_message
            eos_token_id = self.tokenizer.convert_tokens_to_ids(template.sep.strip())

            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            for num_patches in num_patches_list:
                image_tokens = "<img>" + "<IMG_CONTEXT>" * self.model.num_image_token * num_patches + "</img>"
                query = query.replace("<image>", image_tokens, 1)
            
            model_inputs = self.tokenizer(query, return_tensors="pt")
            input_ids = model_inputs["input_ids"].to(self.device)
            attention_mask = model_inputs["attention_mask"].to(self.device)
            generation_config["eos_token_id"] = eos_token_id
        else:
            raise NotImplementedError
        if in_query:
            marker_pos = find_marker_pos(input_ids, self.tokenizer)
            if marker_pos is not None:
                self.set_in_query_mask(attention_mask, marker_pos)
        if marker_only and not_marker:
            raise ValueError("marker_only and not_marker cannot both be True")
        if marker_only:
            from utils.msteer_sample import msteer_sample
            transformers.generation.utils.GenerationMixin._sample = msteer_sample
            self.model._msteer_marker_only = True
            self.model._msteer_not_marker = False
            generation_config.update({"tokenizer": self.tokenizer})
        elif not_marker:
            from utils.msteer_sample import msteer_sample
            transformers.generation.utils.GenerationMixin._sample = msteer_sample
            for layer in self.model.language_model.model.layers:
                layer.in_decode_steer_flag = True
            self.model._msteer_marker_only = False
            self.model._msteer_not_marker = True
            generation_config.update({"tokenizer": self.tokenizer})
        else:
            self.model._msteer_marker_only = False
            self.model._msteer_not_marker = False
            for layer in self.model.language_model.model.layers:
                layer.in_decode_steer_flag = True

        assert len(input_ids.shape) == 2
        return self.generate(pixel_values=pixel_values, tokens=input_ids, attention_mask=attention_mask, **generation_config)

    def get_input_embeds(self, pixel_values, input_ids, visual_features=None):
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                with t.no_grad():
                    vit_embeds = self.model.extract_feature(pixel_values)
                    input_embeds = self.model.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.model.img_context_token_id)
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.model.language_model.get_input_embeddings()(input_ids)
        return input_embeds

    def get_logits(self, image_or_video_path, vision_type, tokens):
        with t.no_grad():
            instr_pos = find_instruction_end_postion(tokens[0], self.END_STR)
            self.set_from_positions(instr_pos)
            if vision_type == "image":
                pixel_values = self.get_pixel_values(image_or_video_path)
            elif vision_type == "video":
                pixel_values, _ = self.get_pixel_values_videos(image_or_video_path)
            else:
                raise NotImplementedError
            input_embeds = self.get_input_embeds(pixel_values, tokens)
            logits = self.model.language_model(inputs_embeds=input_embeds).logits
            return logits

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
        elif type == 'inst_it_image_oe_qa':
            assert 'pil_image' in dataloader_item
            image = dataloader_item['pil_image'].convert('RGB')
            pre = prompt_template["image_oe"]["pre_prompt"]
            post = prompt_template["image_oe"]["post_prompt"]
            prompt = f"{pre}\n{post}\n{dataloader_item['question']}"
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
            return video_path, prompt
        elif type == 'inst_it_video_oe_qa':
            video_root = f"../DATA/Inst-It-Bench/{dataloader_item['video_path']}"
            frames = os.listdir(video_root)
            frames = sorted(frames)
            video_path = [os.path.join(video_root, frame) for frame in frames]
            
            pre = prompt_template["video_oe"]["pre_prompt"]
            post = prompt_template["video_oe"]["post_prompt"]
            
            prompt = f"{pre}\n{post}\n{dataloader_item['question']}"
            return video_path, prompt
        elif type == 'gar_image_mc_qa':
            assert 'pil_image' in dataloader_item
            image = dataloader_item['pil_image'].convert('RGB')
            pre  = prompt_template["image_mc"]["pre_prompt"]
            post = prompt_template["image_mc"]["post_prompt"]
            options = "\n".join(dataloader_item['choices'])
            prompt = f"{pre}\n{dataloader_item['question']}\n{options}\n{post}"
            prompt = re.sub(r'<Prompt(\d+)>', r'[\1]', prompt)
            if not getattr(self, "noref", False):
                image = overlay_visual_prompt(image, dataloader_item["mask_rles"], format="boxnum")
            return image, prompt
        elif type == 'gar_image_simple_oe_qa':
            image_file = os.path.join(dataloader_item['image_path'])
            image = Image.open(image_file).convert('RGB')
            pre = prompt_template["image_oe"]["pre_prompt"]
            post = prompt_template["image_oe"]["post_prompt"]
            prompt = f"{pre}\n{post}\n{dataloader_item['question']}"
            prompt = re.sub(r'<Prompt(\d+)>', r'[\1]', prompt)
            if not getattr(self, "noref", False):
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
        else:
            raise ValueError(f'{type} not supported.')
        return image, prompt
    
    def get_tokens_for_compare(self, image_or_video_path, vision_type, instruction, model_output, IMG_START_TOKEN='<img>', 
        IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>', verbose=False
    ):
        if vision_type == "image":
            pixel_values = self.get_pixel_values(image_or_video_path)
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
            instruction = INST_IT_IMAGE_SYSTEM_PROMPT + '<image>\n' + f"{instruction}"
        elif vision_type == "video":
            pixel_values, num_patches_list = self.get_pixel_values_videos(image_or_video_path)
            instruction = INST_IT_VIDEO_SYSTEM_PROMPT + ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))]) + f"{instruction}"
        else:
            raise NotImplementedError
        
        img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.model.img_context_token_id = img_context_token_id

        template = get_conv_template(self.model.template)
        template.system_message = self.model.system_message

        template.append_message(template.roles[0], instruction)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose:
            print(f"[q]: \033[31m{query}\033[0m")

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.model.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
        
        if model_output is not None:
            query += f" {model_output.strip()}"

        return self.tokenizer.encode(query)
    
    def get_pixel_values(self, image_path_or_image, max_num=12, input_size=448):
        if isinstance(image_path_or_image, str):
            img = Image.open(image_path_or_image).convert('RGB')
        elif isinstance(image_path_or_image, Image.Image):
            img = image_path_or_image
        else:
            raise TypeError
        transform = self.build_transform(input_size=input_size)
        tiles = self.dynamic_preprocess(img, image_size=input_size, 
            use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(im) for im in tiles]
        return t.stack(pixel_values).cuda().to(t.bfloat16)
    
    def get_pixel_values_videos(self, video_path_or_video, max_num=12, input_size=448):
        return internvl_load_frames(video_path_or_video)
    
    def build_transform(self, input_size):
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
        transform = T.Compose(
            [
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD)
            ]
        )
        return transform

    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def get_last_activations(self, layer):
        return self.model.language_model.model.layers[layer].activations

    def set_add_activations(self, layer, activations):
        self.model.language_model.model.layers[layer].add(activations)

    def reset_all(self):
        self.model._msteer_marker_only = False
        self.model._msteer_not_marker = False
        for layer in self.model.language_model.model.layers:
            layer.reset()
