import os
import re
import numpy as np
import pycocotools.mask as maskUtils
import torch
import torchvision.transforms as T

from PIL import Image
from utils.overlay import overlay_xyxy_box, overlay_number, overlay_xywh_box
from typing import List, Tuple, Union
from .load_image import build_transform, dynamic_preprocess

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

colors = ['red', 'green', 'yellow', 'blue', 'grey', 'purple', 'orange', 'brown', 'pink']

def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices

def mask_to_bbox(mask: np.ndarray) -> Union[Tuple[int, int, int, int], None]:
    """
    Extract axis-aligned bounding box from a binary mask.

    Args:
        mask: Binary array of 0s and 1s.

    Returns:
        (x_min, y_min, x_max, y_max) or None if no foreground (1) pixels.
    """
    coords = np.where(mask == 1)
    
    if len(coords[0]) == 0:
        return None
    
    # Min/max coordinates of foreground
    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()
    
    return [x_min, y_min, x_max, y_max]

def load_video(type, item, bound=None, input_size=448, max_num=1, num_segments=32):
    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    if type == 'inst_it_video_mc_qa':
        video_path = f"datasets/test/inst_it/{item['video_path']}"
        frames = os.listdir(video_path)
        frames = sorted(frames)
        frame_id_list = [int(os.path.splitext(frame)[0]) for frame in frames]
        frame_id_list = {i: frame for i, frame in enumerate(frames)}
        question = f"{item['question']}\n"
        for option in item['options']:
            option_str = f"({option}) {item['options'][option]}\n"
            question += option_str
        matches = set(re.findall(r'[\d+]', question))
        anns = item['annotations']
        annotated_images = []
        for i in frame_id_list:
            image = Image.open(os.path.join(video_path, frames[i]))
            frame = frame_id_list[i]
            if frame in anns:
                ann = anns[frame]
                for ins_id in matches:
                    if ins_id in ann:
                        bbox = ann[ins_id]['bbox']
                        image = overlay_xywh_box(image, bbox, color=colors[int(ins_id)] if int(ins_id) < 9 else 'red')
            image = overlay_number(image, i)
            annotated_images.append(image)
        for frame_id, annotated_image in zip(frame_id_list, annotated_images):
            annotated_image.save(f'{frame_id}.png')
        color_str = ''
        for i, obj in enumerate(matches):
            if i != 0:
                if i < 9:
                    color_str += f'; [{obj}]: {colors[i]}'
                else:
                    color_str += f'; [{obj}]: red'
            else:
                color_str += f'[{obj}]: {colors[i]}'
        prompt = f"I have highlighted objects using colored boxes and highlighted the video with red numbers in the lowerleft corner of every frame. For objects, {color_str}. Frames are represented with <frame> in prompt (for example, <4>). {question} Answer with the best option."
    elif type == 'inst_it_video_oe_qa':
        video_path = f"datasets/test/inst_it/{item['video_path']}"
        frames = os.listdir(video_path)
        frames = sorted(frames)
        frame_id_list = [int(os.path.splitext(frame)[0]) for frame in frames]
        frame_id_list = {i: frame for i, frame in enumerate(frames)}
        question = item['question']
        matches = set(re.findall(r'[\d+]', question))
        anns = item['annotations']
        annotated_images = []
        for i in frame_id_list:
            image = Image.open(os.path.join(video_path, frames[i]))
            frame = frame_id_list[i]
            if frame in anns:
                ann = anns[frame]
                for ins_id in matches:
                    if ins_id in ann:
                        bbox = ann[ins_id]['bbox']
                        image = overlay_xywh_box(image, bbox, color=colors[int(ins_id)] if int(ins_id) < 9 else 'red')
            image = overlay_number(image, i)
            annotated_images.append(image)
        color_str = ''
        for i, obj in enumerate(matches):
            if i != 0:
                if i < 9:
                    color_str += f'; [{obj}]: {colors[i]}'
                else:
                    color_str += f'; [{obj}]: red'
            else:
                color_str += f'[{obj}]: {colors[i]}'
        prompt = f"I have highlighted objects using colored boxes and highlighted the video with red numbers in the lowerleft corner of every frame. For objects, {color_str}. Frames are represented with <frame> in prompt (for example, <4>). {question}"
    else:
        annotated_images = None
        raise ValueError(f'{type} not supported.')
    for annotated_image in annotated_images:
        img = dynamic_preprocess(annotated_image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, prompt

def frame_sample(duration, video_frames=16):
    num_frames = 32
    seg_size = float(duration - 1) / num_frames

    raw_frame_ids = []
    for i in range(num_frames):
        start = int(np.round(seg_size * i))
        end = int(np.round(seg_size * (i + 1)))
        raw_frame_ids.append((start + end) // 2)

    sampled_frame_ids = []
    seg_size = float(num_frames - 1) / video_frames
    for i in range(video_frames):
        start = int(np.round(seg_size * i))
        end = int(np.round(seg_size * (i + 1)))
        sampled_frame_ids.append(raw_frame_ids[(start + end) // 2])
    return sampled_frame_ids

def annToMask(rle):
    m = maskUtils.decode(rle)
    return m