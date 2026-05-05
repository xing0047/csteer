import numpy as np
from pycocotools import mask as mask_utils
from PIL import ImageDraw, ImageFont

COLOR = ["red", "blue", "green"]


def overlay_visual_prompt(image, mask_rles, format="boxnum", box_width=2):
    if format != "boxnum":
        raise NotImplementedError(f"Only format='boxnum' is supported, got {format!r}")
    assert mask_rles is not None
    boxs = []
    for mask_id, mask_rle in enumerate(mask_rles):
        mask_np = mask_utils.decode(mask_rle).astype(np.uint8)
        mask_np = (mask_np * 255).astype(np.uint8)
        boxs.append(mask_to_box(mask_np))

    image_copy = image.copy()
    draw = ImageDraw.Draw(image_copy)
    img_w, img_h = image_copy.size

    for box_id, box in enumerate(boxs):
        x1, y1, x2, y2 = box
        color = COLOR[box_id % len(COLOR)]

        draw.rectangle([x1, y1, x2, y2], outline=color, width=box_width)

        text = str(box_id)

        try:
            font = ImageFont.truetype("arial.ttf", size=80)
        except OSError:
            font = ImageFont.load_default()

        bbox = draw.textbbox((x1, y1), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        bg_x1 = x1
        bg_y1 = y1 - text_height - 8

        if bg_y1 < 0:
            bg_y1 = y1

        bg_x2 = bg_x1 + text_width + 8
        bg_y2 = bg_y1 + text_height + 8

        if bg_x2 > img_w:
            shift = bg_x2 - img_w
            bg_x1 = max(0, bg_x1 - shift)
            bg_x2 = bg_x1 + text_width + 8

        bg_y1 = max(0, min(bg_y1, img_h - text_height - 8))
        bg_y2 = bg_y1 + text_height + 8

        background_box = [bg_x1, bg_y1, bg_x2, bg_y2]
        draw.rectangle(background_box, fill=color)

        text_x = bg_x1 + 4
        text_y = bg_y1 + 4
        draw.text((text_x, text_y), text, fill=(255, 255, 255), font=font)
    return image_copy


def mask_to_box(mask, format="xyxy"):
    rows, cols = np.where(mask > 0)

    if len(rows) == 0:
        return None

    if format == "xyxy":
        y1, y2 = rows.min(), rows.max()
        x1, x2 = cols.min(), cols.max()
        return (x1, y1, x2, y2)
    raise NotImplementedError
