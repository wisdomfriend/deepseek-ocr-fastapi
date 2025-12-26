"""
图片处理工具函数
"""
import re
from typing import List, Tuple, Optional
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps
import numpy as np

from app.core.config import RESOLUTION_CONFIGS


def load_image(image_path: Path) -> Optional[Image.Image]:
    """加载并处理图片"""
    try:
        image = Image.open(image_path)
        corrected_image = ImageOps.exif_transpose(image)
        return corrected_image.convert('RGB')
    except Exception as e:
        print(f"Error loading image: {e}")
        return None


def re_match(text: str):
    """解析OCR结果中的边界框信息"""
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(pattern, text, re.DOTALL)
    
    matches_image = []
    matches_other = []
    for a_match in matches:
        if '<|ref|>image<|/ref|>' in a_match[0]:
            matches_image.append(a_match[0])
        else:
            matches_other.append(a_match[0])
    return matches, matches_image, matches_other


def parse_blocks_with_text(full_text: str):
    """将 DeepSeek 段落解析为 [ {label, points_list, content} ]
    规则：每个 <|ref|>..</|ref|><|det|>..</|det|> 之后紧跟的自然语言文字（直到下一个 ref 块开始）作为该框的 content。
    """
    blocks = []
    pattern = re.compile(r'<\|ref\|>(?P<label>.*?)<\|/ref\|><\|det\|>(?P<det>.*?)<\|/det\|>', re.DOTALL)
    iters = list(pattern.finditer(full_text))
    for idx, m in enumerate(iters):
        label = m.group('label').strip()
        det = m.group('det').strip()
        # points_list: [[x1,y1,x2,y2], ...]
        try:
            points_list = eval(det)
        except Exception:
            points_list = []
        # content slice
        start = m.end()
        end = iters[idx + 1].start() if idx + 1 < len(iters) else len(full_text)
        content = full_text[start:end].strip()
        # 只取第一行非空作为该框的主要文本描述
        first_line = ''
        for line in content.splitlines():
            s = line.strip()
            if s:
                first_line = s
                break
        blocks.append({
            'label': label,
            'points_list': points_list if isinstance(points_list, list) else [],
            'content': first_line
        })
    return blocks


def extract_coordinates_and_label(ref_text, image_width, image_height):
    """提取坐标和标签信息"""
    try:
        label_type = ref_text[1]
        cor_list = eval(ref_text[2])
        return (label_type, cor_list)
    except Exception as e:
        print(f"Error extracting coordinates: {e}")
        return None


def draw_bounding_boxes(image: Image.Image, refs: List, output_path: Path):
    """绘制边界框"""
    image_width, image_height = image.size
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    
    overlay = Image.new('RGBA', img_draw.size, (0, 0, 0, 0))
    draw2 = ImageDraw.Draw(overlay)
    
    font = ImageFont.load_default()
    img_idx = 0
    
    # 创建images目录
    images_dir = output_path.parent / "images"
    images_dir.mkdir(exist_ok=True)
    
    for i, ref in enumerate(refs):
        try:
            result = extract_coordinates_and_label(ref, image_width, image_height)
            if result:
                label_type, points_list = result
                
                color = (np.random.randint(0, 200), np.random.randint(0, 200), np.random.randint(0, 255))
                color_a = color + (20, )
                
                for points in points_list:
                    x1, y1, x2, y2 = points
                    
                    x1 = int(x1 / 999 * image_width)
                    y1 = int(y1 / 999 * image_height)
                    x2 = int(x2 / 999 * image_width)
                    y2 = int(y2 / 999 * image_height)
                    
                    if label_type == 'image':
                        try:
                            cropped = image.crop((x1, y1, x2, y2))
                            cropped.save(images_dir / f"{img_idx}.jpg")
                            img_idx += 1
                        except Exception as e:
                            print(f"Error saving cropped image: {e}")
                    
                    try:
                        if label_type == 'title':
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
                            draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)
                        else:
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                            draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)
                        
                        text_x = x1
                        text_y = max(0, y1 - 15)
                        
                        text_bbox = draw.textbbox((0, 0), label_type, font=font)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]
                        draw.rectangle([text_x, text_y, text_x + text_width, text_y + text_height],
                                      fill=(255, 255, 255, 30))
                        
                        draw.text((text_x, text_y), label_type, font=font, fill=color)
                    except Exception as e:
                        print(f"Error drawing text: {e}")
        except Exception as e:
            print(f"Error processing ref: {e}")
            continue
    
    img_draw.paste(overlay, (0, 0), overlay)
    return img_draw


def convert_matches_to_results(matches_ref, image_width: int, image_height: int, contents: list = None):
    """将 DeepSeek 解析到的矩形区域转换为 ocr_server 的 OCRResult 列表格式。
    bbox 使用四点坐标顺时针：[ [x1,y1], [x2,y1], [x2,y2], [x1,y2] ]。
    置信度暂设为 1.0。
    文本使用内容优先（content），若无则回退到标签 label。
    """
    results = []
    if not matches_ref:
        return results
    for idx, ref in enumerate(matches_ref):
        try:
            extracted = extract_coordinates_and_label(ref, image_width, image_height)
            if not extracted:
                continue
            label_type, points_list = extracted
            box_text = None
            if contents and idx < len(contents):
                box_text = contents[idx]
            for points in points_list:
                x1, y1, x2, y2 = points
                # DeepSeek 坐标是 0..999 归一化，需要还原到像素尺寸
                x1 = int(x1 / 999 * image_width)
                y1 = int(y1 / 999 * image_height)
                x2 = int(x2 / 999 * image_width)
                y2 = int(y2 / 999 * image_height)
                bbox = [
                    [float(x1), float(y1)],
                    [float(x2), float(y1)],
                    [float(x2), float(y2)],
                    [float(x1), float(y2)]
                ]
                results.append({
                    "label": label_type,
                    "text": box_text or label_type,
                    "confidence": 1.0,
                    "bbox": bbox
                })
        except Exception:
            continue
    return results

