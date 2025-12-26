"""
OCR 相关端点
"""
import uuid
import time
import logging
import asyncio
import os
from typing import Optional
from pathlib import Path
from fastapi import UploadFile, File, Form, HTTPException, BackgroundTasks

from app.models.schemas import OCRResponse, TaskStatus
from app.services.ocr_service import process_ocr_task
from app.core.config import (
    UPLOAD_DIR, ALLOWED_EXTENSIONS, RESOLUTION_CONFIGS, TASK_PROMPTS, MAX_CONCURRENCY
)

logger = logging.getLogger(__name__)

# 任务存储（生产环境应使用Redis等）
tasks = {}

# 并发控制：限制同时执行的OCR任务数量
# 默认使用配置中的 MAX_CONCURRENCY，但考虑到GPU内存限制，建议设置为1-4
# 这个信号量确保不会同时处理过多任务导致GPU内存溢出
MAX_CONCURRENT_OCR_TASKS = int(os.getenv("MAX_CONCURRENT_OCR_TASKS", str(min(MAX_CONCURRENCY, 3))))
task_semaphore = asyncio.Semaphore(MAX_CONCURRENT_OCR_TASKS)

logger.info(f"OCR任务并发控制：最多同时执行 {MAX_CONCURRENT_OCR_TASKS} 个任务")


async def upload_and_process(
    file: UploadFile = File(...),
    include_visualization: bool = Form(True),
    resolution: str = Form("gundam"),
    task_type: str = Form("markdown"),
    reference_text: Optional[str] = Form(None),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """上传图片并进行OCR处理
    
    任务处理机制说明：
    1. 接收文件并保存到磁盘
    2. 创建任务记录（状态：pending）
    3. 将处理函数添加到 FastAPI 的 BackgroundTasks 队列
    4. 立即返回任务ID，不等待处理完成
    5. FastAPI 在响应返回后自动执行后台任务
    6. 客户端通过 /api/tasks/{task_id} 轮询任务状态
    
    注意：BackgroundTasks 在响应返回后执行，适合短时间任务。
    生产环境建议使用 Redis + Celery 实现真正的任务队列。
    """
    
    # 检查文件类型
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"不支持的文件格式: {file_ext}")
    
    # 检查分辨率配置
    if resolution not in RESOLUTION_CONFIGS:
        raise HTTPException(status_code=400, detail=f"不支持的分辨率: {resolution}")
    
    # 检查任务类型
    if task_type not in TASK_PROMPTS and not task_type.startswith("<"):
        raise HTTPException(status_code=400, detail=f"不支持的任务类型: {task_type}")
    
    # 生成任务ID
    task_id = str(uuid.uuid4())
    
    # 保存上传的文件
    upload_path = UPLOAD_DIR / f"{task_id}{file_ext}"
    with open(upload_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    
    # 创建任务
    tasks[task_id] = TaskStatus(
        task_id=task_id,
        status="pending",
        created_at=time.time()
    )
    
    # 添加后台任务
    background_tasks.add_task(
        _process_task,
        task_id,
        upload_path,
        resolution,
        task_type,
        reference_text,
        include_visualization
    )
    
    return OCRResponse(
        task_id=task_id,
        status="pending"
    )


async def _process_task(
    task_id: str,
    image_path: Path,
    resolution: str,
    task_type: str,
    reference_text: Optional[str],
    include_visualization: bool
):
    """处理任务的内部函数
    
    此函数由 FastAPI 的 BackgroundTasks 在响应返回后自动调用。
    
    并发控制说明：
    - 使用 asyncio.Semaphore 限制同时执行的任务数
    - 如果1000个请求同时到达：
      * 所有请求会立即返回任务ID（不阻塞）
      * 但只有 MAX_CONCURRENT_OCR_TASKS 个任务会同时执行
      * 其他任务会排队等待，直到有任务完成释放信号量
    
    执行流程：
    1. 获取信号量（如果已满则等待）
    2. 更新任务状态为 "processing"
    3. 调用服务层的 process_ocr_task 进行实际OCR处理
    4. 根据处理结果更新任务状态（completed 或 failed）
    5. 释放信号量（允许下一个任务执行）
    """
    # 使用信号量控制并发：如果已经有 MAX_CONCURRENT_OCR_TASKS 个任务在执行，
    # 这里会等待，直到有任务完成并释放信号量
    async with task_semaphore:
        try:
            logger.info(f"开始处理任务 {task_id} (当前并发: {MAX_CONCURRENT_OCR_TASKS - task_semaphore._value})")
            tasks[task_id].status = "processing"
            result = await process_ocr_task(
                task_id, image_path, resolution, task_type, reference_text, include_visualization
            )
            tasks[task_id].status = "completed"
            tasks[task_id].result = result
            tasks[task_id].completed_at = time.time()
            logger.info(f"任务 {task_id} 处理完成")
        except Exception as e:
            logger.exception(f"Error processing task {task_id}: {e}")
            tasks[task_id].status = "failed"
            tasks[task_id].error = str(e)
            tasks[task_id].completed_at = time.time()
            logger.error(f"任务 {task_id} 处理失败: {str(e)}")


async def get_task_status(task_id: str):
    """获取任务状态"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    return tasks[task_id]


async def list_tasks():
    """列出所有任务"""
    return list(tasks.values())


async def binary_ocr_endpoint(
    image_data: UploadFile = File(...),
    height: int = Form(...),
    width: int = Form(...),
    task_type: str = Form("markdown"),
    resolution: str = Form("gundam")
):
    """与 ocr_server 格式对齐的二进制 OCR 接口，但使用 DeepSeek 本地推理。"""
    try:
        from app.services.ocr_service import run_deepseek_on_pil
        import numpy as np
        from PIL import Image
        
        # 读取并还原为 RGB 图
        start_time = time.time()
        image_bytes = await image_data.read()
        img_array = np.frombuffer(image_bytes, dtype=np.uint8).reshape(height, width, 3)
        pil_image = Image.fromarray(img_array, mode='RGB')

        text, processed_text, results = await run_deepseek_on_pil(pil_image, task_type, resolution)
        elapsed = time.time() - start_time

        return {
            "success": True,
            "results": results,
            "processing_time": round(elapsed, 4),
            "image_size": {"width": width, "height": height},
            "text": text,
            "processed_text": processed_text
        }
    except Exception as e:
        logger.exception(f"Binary OCR error: {e}")
        raise HTTPException(status_code=500, detail=f"binary_ocr error: {str(e)}")

