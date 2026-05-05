import numpy as np
from pycocotools import mask as mask_utils
from PIL import Image, ImageDraw, ImageFont
COLOR = ["red", "blue", "green"]
def overlay_visual_prompt(image, mask_rles, format="box", box_width=2):
    if format == "box":
        raise NotImplementedError
    elif format == "boxnum":
        assert mask_rles is not None
        boxs = []
        for mask_id, mask_rle in enumerate(mask_rles):
            mask_np = mask_utils.decode(mask_rle).astype(np.uint8)
            mask_np = (mask_np * 255).astype(np.uint8)
            boxs.append(mask_to_box(mask_np))
        # draw
        image_copy = image.copy()
        draw = ImageDraw.Draw(image_copy)
        img_w, img_h = image_copy.size

        for box_id, box in enumerate(boxs):
            x1, y1, x2, y2 = box
            color = COLOR[box_id % len(COLOR)]
            
            # 绘制矩形
            draw.rectangle([x1, y1, x2, y2], outline=color, width=box_width)
            
            # 绘制编号文字
            text = str(box_id)  # 或者 str(box_id) 如果你想从0开始
            
            # 尝试使用更大的字体，如果失败则使用默认字体
            try:
                font = ImageFont.truetype("arial.ttf", size=80)  # 可以调整size
            except:
                font = ImageFont.load_default()
            
            # 获取文字边界框
            bbox = draw.textbbox((x1, y1), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # 默认画在框的上方左侧
            bg_x1 = x1
            bg_y1 = y1 - text_height - 8

            # 如果上方空间不足，则把数字框放到框内（靠近左上）
            if bg_y1 < 0:
                bg_y1 = y1  # 改为从框的顶部开始向下画

            bg_x2 = bg_x1 + text_width + 8
            bg_y2 = bg_y1 + text_height + 8

            # 右侧越界时向左收缩，保证在图像内
            if bg_x2 > img_w:
                shift = bg_x2 - img_w
                bg_x1 = max(0, bg_x1 - shift)
                bg_x2 = bg_x1 + text_width + 8

            # 再次确保不越界（防御性判断）
            bg_y1 = max(0, min(bg_y1, img_h - text_height - 8))
            bg_y2 = bg_y1 + text_height + 8
            
            # 绘制文字背景（让文字更清晰）
            background_box = [bg_x1, bg_y1, bg_x2, bg_y2]
            draw.rectangle(background_box, fill=color)
            
            # 绘制文字（白色），稍微留一点内边距
            text_x = bg_x1 + 4
            text_y = bg_y1 + 4
            draw.text((text_x, text_y), text, fill=(255, 255, 255), font=font)
        return image_copy
    else:
        raise NotImplementedError

def mask_to_box(mask, format='xyxy'):
    rows, cols = np.where(mask > 0)

    if len(rows) == 0:
        return None

    if format == "xyxy":
        y1, y2 = rows.min(), rows.max()
        x1, x2 = cols.min(), cols.max()

        return (x1, y1, x2, y2)
    else:
        raise NotImplementedError

def overlay_xyxy_box(image, xyxy, color, width=2):
    """
    Overlay a bounding box on a PIL image given XYXY coordinates.
    
    Args:
        image (PIL.Image): Input PIL image
        xyxy (tuple or list): Bounding box coordinates in format (x1, y1, x2, y2)
                             where (x1, y1) is top-left and (x2, y2) is bottom-right
        color (str or tuple): Color for the bounding box. Can be:
                             - String: 'red', 'blue', 'green', etc.
                             - RGB tuple: (255, 0, 0) for red
                             - RGBA tuple: (255, 0, 0, 128) for semi-transparent red
        width (int): Width of the bounding box lines in pixels (default: 2)
    
    Returns:
        PIL.Image: Image with bounding box overlaid
    """
    if xyxy is None:
        return image
    # Create a copy to avoid modifying the original image
    img_copy = image.copy()
    
    # Create drawing context
    draw = ImageDraw.Draw(img_copy)
    
    # Extract coordinates
    x1, y1, x2, y2 = xyxy
    
    # Draw the rectangle
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    
    return img_copy

def overlay_xywh_box(image, xywh, color, width=2):
    """
    Overlay a bounding box on a PIL image given XYXY coordinates.
    
    Args:
        image (PIL.Image): Input PIL image
        xywh (tuple or list): Bounding box coordinates in format (x, y, w, h)
                             where (x, y) is top-left and (w, h) is width, height
        color (str or tuple): Color for the bounding box. Can be:
                             - String: 'red', 'blue', 'green', etc.
                             - RGB tuple: (255, 0, 0) for red
                             - RGBA tuple: (255, 0, 0, 128) for semi-transparent red
        width (int): Width of the bounding box lines in pixels (default: 2)
    
    Returns:
        PIL.Image: Image with bounding box overlaid
    """
    if xywh is None:
        return image
    # Create a copy to avoid modifying the original image
    img_copy = image.copy()
    
    # Create drawing context
    draw = ImageDraw.Draw(img_copy)
    
    # Extract coordinates
    x1, y1, x2, y2 = xywh[0], xywh[1], xywh[0] + xywh[2], xywh[1] + xywh[3]
    
    # Draw the rectangle
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    
    return img_copy

def overlay_number(image, number, color='red', font_size=40):
    """
    Overlay a number on a PIL image at the bottom-left corner.
   
    Args:
        image (PIL.Image): Input PIL image
        number (int or str): Number to overlay on the image
        color (str or tuple): Color for the number. Can be:
                             - String: 'red', 'blue', 'green', etc.
                             - RGB tuple: (255, 0, 0) for red
                             - RGBA tuple: (255, 0, 0, 128) for semi-transparent red
        font_size (int): Font size in pixels (default: 40)
   
    Returns:
        PIL.Image: Image with number overlaid at bottom-left corner
    """
    from PIL import ImageDraw, ImageFont
    
    if number is None:
        return image
    
    # Create a copy to avoid modifying the original image
    img_copy = image.copy()
   
    # Create drawing context
    draw = ImageDraw.Draw(img_copy)
    
    # Convert number to string
    number_str = str(number)
    
    # Try to load a better font, fall back to default if not available
    try:
        # On Windows
        font = ImageFont.truetype("arial.ttf", font_size)
    except (OSError, IOError):
        try:
            # On macOS
            font = ImageFont.truetype("Arial.ttf", font_size)
        except (OSError, IOError):
            try:
                # On Linux
                font = ImageFont.truetype("DejaVuSans.ttf", font_size)
            except (OSError, IOError):
                # Fall back to default font
                font = ImageFont.load_default()
    
    # Get text dimensions
    bbox = draw.textbbox((0, 0), number_str, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Get image dimensions
    img_width, img_height = image.size
    
    # Calculate position for bottom-left corner with some padding
    padding = 10
    x = padding
    y = img_height - text_height - padding
    
    # Draw the number
    draw.text((x, y), number_str, fill=color, font=font)
   
    return img_copy
