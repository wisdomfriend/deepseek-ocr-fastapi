"""
OCR 业务逻辑服务
"""
import time
import logging
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
from io import BytesIO
import sys

import numpy as np
from PIL import Image
from vllm import SamplingParams

# 添加DeepSeek-OCR-vllm到路径
sys.path.append('/app/DeepSeek-OCR-vllm')

from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from app.core.lifespan import get_engine, get_processor
from app.core.config import (
    RESOLUTION_CONFIGS, TASK_PROMPTS, OUTPUT_DIR, BASE_SIZE, IMAGE_SIZE, CROP_MODE
)
from app.utils.image_utils import (
    load_image, re_match, parse_blocks_with_text, draw_bounding_boxes,
    convert_matches_to_results
)

logger = logging.getLogger(__name__)


async def stream_generate(image=None, prompt=''):
    """使用全局引擎进行推理"""
    engine = get_engine()
    
    if engine is None:
        raise Exception("Engine not initialized. Please restart the service.")
    
    logits_processors = [NoRepeatNGramLogitsProcessor(
        ngram_size=30, 
        window_size=90, 
        whitelist_token_ids={128821, 128822}
    )]  # whitelist: <td>, </td>

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=8192,
        logits_processors=logits_processors,
        skip_special_tokens=False,
    )

    request_id = f"request-{int(time.time())}"

    printed_length = 0

    if image and '<image>' in prompt:
        request = {
            "prompt": prompt,
            "multi_modal_data": {"image": image}
        }
    elif prompt:
        request = {
            "prompt": prompt
        }
    else:
        raise ValueError('prompt is none!!!')
    
    async for request_output in engine.generate(
        request, sampling_params, request_id
    ):
        if request_output.outputs:
            full_text = request_output.outputs[0].text
            new_text = full_text[printed_length:]
            print(new_text, end='', flush=True)
            printed_length = len(full_text)
            final_output = full_text
    print('\n')

    return final_output


async def process_ocr_task(
    task_id: str, 
    image_path: Path, 
    resolution: str, 
    task_type: str, 
    reference_text: Optional[str], 
    include_visualization: bool = True
):
    """处理OCR任务 - 支持动态配置"""
    try:
        # 加载图片
        image = load_image(image_path)
        if image is None:
            raise Exception("Failed to load image")
        
        # 根据任务类型生成提示词
        if task_type == "locate_object" and reference_text:
            prompt = TASK_PROMPTS[task_type].format(reference_text=reference_text)
        elif task_type in TASK_PROMPTS:
            prompt = TASK_PROMPTS[task_type]
        else:
            # 将 task_type 作为自定义提示词使用
            prompt = task_type
        
        # 根据分辨率配置动态调整参数
        resolution_config = RESOLUTION_CONFIGS.get(resolution, RESOLUTION_CONFIGS["gundam"])
        
        processor = get_processor()
        if processor is None:
            raise Exception("Processor not initialized")
        
        # 临时更新 processor 的实例变量以使用新的分辨率配置
        original_image_size = processor.image_size
        original_base_size = processor.base_size
        
        processor.image_size = resolution_config["image_size"]
        processor.base_size = resolution_config["base_size"]
        crop_mode = resolution_config["crop_mode"]
        
        try:
            # 处理图片
            if '<image>' in prompt:
                image_features = processor.tokenize_with_images(
                    images=[image], 
                    bos=True, 
                    eos=True, 
                    cropping=crop_mode
                )
            else:
                image_features = ''
            
            # 调用stream_generate
            result_out = await stream_generate(image_features, prompt)
            
        finally:
            # 恢复原始配置
            processor.image_size = original_image_size
            processor.base_size = original_base_size
        
        # 处理结果
        result = {
            "text": result_out,
            "image_path": str(image_path),
            "prompt": prompt,
            "resolution": resolution,
            "task_type": task_type
        }
        
        # 生成可视化结果
        if include_visualization and '<image>' in prompt:
            output_path = OUTPUT_DIR / f"{task_id}"
            output_path.mkdir(exist_ok=True)
            
            # 保存原始结果
            with open(output_path / "result_ori.mmd", 'w', encoding='utf-8') as f:
                f.write(result_out)
            
            # 解析边界框
            matches_ref, matches_images, matches_other = re_match(result_out)
            
            # 绘制边界框
            if matches_ref:
                result_image = draw_bounding_boxes(image, matches_ref, output_path)
                result_image.save(output_path / "result_with_boxes.jpg")
                result["visualization_path"] = f"/deepseek-ocr/outputs/{task_id}/result_with_boxes.jpg"
            
            # 处理markdown结果
            processed_text = result_out
            for idx, a_match_image in enumerate(matches_images):
                processed_text = processed_text.replace(
                    a_match_image, 
                    f'![](images/{idx}.jpg)\n'
                )
            
            for a_match_other in matches_other:
                processed_text = processed_text.replace(
                    a_match_other, 
                    ''
                ).replace('\\coloneqq', ':=').replace('\\eqqcolon', '=:')
            
            # 保存处理后的结果
            with open(output_path / "result.mmd", 'w', encoding='utf-8') as f:
                f.write(processed_text)
            
            result["processed_text"] = processed_text
            result["markdown_path"] = f"/deepseek-ocr/outputs/{task_id}/result.mmd"
        
        return result
        
    except Exception as e:
        logger.exception(f"Error processing OCR task {task_id}: {e}")
        raise


async def run_deepseek_on_pil(
    image: Image.Image, 
    task_type: str = "markdown", 
    resolution: str = "gundam"
) -> Tuple[str, str, list]:
    """对 PIL.Image 运行 DeepSeek OCR，返回原始文本、处理后文本、矩形结果。"""
    # 生成提示词
    if task_type in TASK_PROMPTS:
        prompt = TASK_PROMPTS[task_type]
    else:
        # 将 task_type 作为自定义提示词使用
        prompt = task_type
    
    # 根据分辨率配置动态调整参数
    resolution = resolution if resolution in RESOLUTION_CONFIGS else "gundam"
    resolution_config = RESOLUTION_CONFIGS[resolution]

    processor = get_processor()
    if processor is None:
        raise Exception("Processor not initialized")
    
    # 临时更新 processor 的实例变量
    original_image_size = processor.image_size
    original_base_size = processor.base_size
    
    processor.image_size = resolution_config["image_size"]
    processor.base_size = resolution_config["base_size"]
    crop_mode = resolution_config["crop_mode"]

    try:
        # 处理图像
        if '<image>' in prompt:
            image_features = processor.tokenize_with_images(
                images=[image], 
                bos=True, 
                eos=True, 
                cropping=crop_mode
            )
        else:
            image_features = ''

        result_out = await stream_generate(image_features, prompt)
    finally:
        # 恢复原始配置
        processor.image_size = original_image_size
        processor.base_size = original_base_size

    # 解析检测框与对应文本
    matches_ref, matches_images, matches_other = re_match(result_out)
    blocks = parse_blocks_with_text(result_out)
    contents = [b.get('content', '') for b in blocks]
    results = convert_matches_to_results(matches_ref, image.size[0], image.size[1], contents)

    # 生成 processed_text（替换图片占位）
    processed_text = result_out
    for idx, a_match_image in enumerate(matches_images):
        processed_text = processed_text.replace(a_match_image, f'![](images/{idx}.jpg)\n')
    for a_match_other in matches_other:
        processed_text = processed_text.replace(a_match_other, '').replace('\\coloneqq', ':=').replace('\\eqqcolon', '=:')

    return result_out, processed_text, results

