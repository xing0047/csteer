import numpy as np
import torch
import re
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from utils.overlay import overlay_xyxy_box

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

colors = ['red', 'green', 'yellow', 'blue', 'grey', 'purple', 'orange', 'brown', 'pink']


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
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


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
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


# def load_image(type, item, input_size=448, max_num=12, box=None, box_color='red'):
#     if type == 'vip_image_oe_qa':
#         image_file = f"datasets/test/vipbench/bbox/images/{item['image']}"
#         image = Image.open(image_file).convert('RGB')
#         prompt = item['text']
#     elif type == 'inst_it_image_mc_qa':
#         image_file = f"datasets/test/inst_it/{item['image_path']}"
#         image = Image.open(image_file).convert('RGB')
#         question = f"{item['question']}\n"
#         for option in item['options']:
#             option_str = f"({option}) {item['options'][option]}\n"
#             question += option_str
#         matches = set(re.findall(r'[\d+]', question))
#         color_str = ''
#         for i, obj in enumerate(matches):
#             if obj in item['annotations']:
#                 bbox = item['annotations'][obj]
#             else:
#                 bbox = None
#             image = overlay_xyxy_box(image, bbox, color=colors[i] if i < 9 else 'red')
#             if i != 0:
#                 if i < 9:
#                     color_str += f'; [{obj}]: {colors[i]}'
#                 else:
#                     color_str += f'; [{obj}]: red'
#             else:
#                 color_str += f'[{obj}]: {colors[i]}'
#         prompt = f'I have highlighted objects using colored boxes in the image. Specifically, {color_str}. {question} Answer with the best option.'
#     elif type == 'inst_it_image_oe_qa':
#         image_file = f"datasets/test/inst_it/{item['image_path']}"
#         image = Image.open(image_file).convert('RGB')
#         question = f"{item['question']}"
#         matches = set(re.findall(r'[\d+]', question))
#         color_str = ''
#         for i, obj in enumerate(matches):
#             if obj in item['annotations']:
#                 bbox = item['annotations'][obj]
#             else:
#                 bbox = None
#             image = overlay_xyxy_box(image, bbox, color=colors[i] if i < 9 else 'red')
#             if i != 0:
#                 if i < 9:
#                     color_str += f'; [{obj}]: {colors[i]}'
#                 else:
#                     color_str += f'; [{obj}]: red'
#             else:
#                 color_str += f'[{obj}]: {colors[i]}'
#         prompt = f'I have highlighted objects using colored boxes in the image. Specifically, {color_str}. {question}'
#     else:
#         raise ValueError(f'{type} not supported.')
#     transform = build_transform(input_size=input_size)
#     images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
#     pixel_values = [transform(image) for image in images]
#     pixel_values = torch.stack(pixel_values)
#     return pixel_values, prompt


def load_image(type, item, input_size=448, max_num=12, box=None, box_color='red'):
    import os
    from utils.prompts import prompt_template
    if type == 'vip_image_oe_qa':
        image_file = f"datasets/test/vipbench/bbox/images/{item['image']}"
        image = Image.open(image_file).convert('RGB')
        prompt = item['text']
    elif type == 'inst_it_image_mc_qa':
        if 'pil_image' in item and isinstance(item['pil_image'], Image.Image):
            image = item['pil_image'].convert('RGB')
        elif 'image' in item and isinstance(item['image'], Image.Image):
            image = item['image'].convert('RGB')
        else:
            image_file = os.path.join("../data/Inst-IT-Dataset", item.get('image_path', ''))
            image = Image.open(image_file).convert('RGB')

        # 2) Official prompt assembly
        pre  = prompt_template["image_mc"]["pre_prompt"]
        post = prompt_template["image_mc"]["post_prompt"]
        options = (
            f"A. {item['choice_a']}\n"
            f"B. {item['choice_b']}\n"
            f"C. {item['choice_c']}\n"
            f"D. {item['choice_d']}"
        )
        prompt = f"{pre}\n{item['question']}\n{options}\n{post}"
    elif type == 'inst_it_image_oe_qa':
        if 'pil_image' in item and isinstance(item['pil_image'], Image.Image):
            image = item['pil_image'].convert('RGB')
        elif 'image' in item and isinstance(item['image'], Image.Image):
            image = item['image'].convert('RGB')
        else:
            image_file = os.path.join("../data/Inst-IT-Dataset", item.get('image_path', ''))
            image = Image.open(image_file).convert('RGB')

        pre  = prompt_template["image_oe"]["pre_prompt"]
        post = prompt_template["image_oe"]["post_prompt"]
        # Official order: pre, post, then question
        prompt = f"{pre}\n{post}\n{item['question']}"
    else:
        raise ValueError(f'{type} not supported.')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values, prompt



def internvl_load_image(path, max_num=12, input_size=448):
    img = Image.open(path).convert('RGB')
    transform = build_transform(input_size=input_size)
    tiles = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(im) for im in tiles]
    return torch.stack(pixel_values)

def internvl_load_frames(video_path, bound=None, input_size=448, max_num=1):
    """
        https://huggingface.co/OpenGVLab/InternVL3-8B#inference-with-transformers
    """
    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    for frame_path in video_path:
        img = Image.open(frame_path).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values).cuda().to(torch.bfloat16)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list