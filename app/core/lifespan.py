"""
应用生命周期管理
"""
import os
import sys
import time
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from vllm import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.model_executor.models.registry import ModelRegistry

# 添加DeepSeek-OCR-vllm到路径
sys.path.append('/app/DeepSeek-OCR-vllm')

from deepseek_ocr import DeepseekOCRForCausalLM
from process.image_process import DeepseekOCRProcessor
from app.core.config import (
    MODEL_PATH, BASE_SIZE, IMAGE_SIZE, CROP_MODE, PROMPT
)

# 设置环境变量
os.environ['VLLM_USE_V1'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("CUDA_VISIBLE_DEVICES", "0")

# 注册模型
ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)

logger = logging.getLogger(__name__)

# 全局变量存储模型引擎和处理器
engine: AsyncLLMEngine = None
processor: DeepseekOCRProcessor = None


def get_engine():
    """获取模型引擎实例"""
    global engine
    return engine


def get_processor():
    """获取处理器实例"""
    global processor
    return processor


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global engine, processor
    
    logger.info("Initializing DeepSeek OCR model...")
    start_time = time.time()
    
    try:
        # 初始化处理器
        processor = DeepseekOCRProcessor()
        logger.info("Processor initialized")
        
        # 初始化引擎
        engine_args = AsyncEngineArgs(
            model=MODEL_PATH,
            hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
            block_size=64,
            max_model_len=8192,
            enforce_eager=False,
            trust_remote_code=True,
            tensor_parallel_size=1,
            gpu_memory_utilization=float(os.getenv("GPU_MEMORY_UTILIZATION", "0.75")),
        )
        
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        logger.info(f"Model engine loaded in {time.time() - start_time:.2f}s")
        
    except Exception as e:
        logger.exception(f"DeepSeek OCR model initialization failed: {str(e)}")
        raise
    
    yield
    
    # 清理资源
    if engine is not None:
        del engine
        engine = None
        logger.info("Model engine resources released")
    
    if processor is not None:
        del processor
        processor = None
        logger.info("Processor resources released")

