"""
应用配置模块
"""
import os
from pathlib import Path

# 模型配置
MODEL_PATH = os.getenv("MODEL_PATH", "/models/DeepSeek-OCR")

# 图片处理配置
BASE_SIZE = int(os.getenv("BASE_SIZE", "1024"))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "640"))
CROP_MODE = os.getenv("CROP_MODE", "true").lower() == "true"
MIN_CROPS = int(os.getenv("MIN_CROPS", "2"))
MAX_CROPS = int(os.getenv("MAX_CROPS", "6"))

# 并发配置
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "10"))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "4"))

# 调试配置
PRINT_NUM_VIS_TOKENS = os.getenv("PRINT_NUM_VIS_TOKENS", "false").lower() == "true"
SKIP_REPEAT = os.getenv("SKIP_REPEAT", "true").lower() == "true"

# 默认提示词
PROMPT = os.getenv("PROMPT", "<image>\n<|grounding|>Convert the document to markdown.")

# 服务配置
HOST = os.getenv("OCR_HOST", "0.0.0.0")
PORT = int(os.getenv("OCR_PORT", "8000"))
LOG_LEVEL = os.getenv("OCR_LOG_LEVEL", "INFO")
RELOAD = os.getenv("OCR_RELOAD", "false").lower() == "true"

# 文件上传配置
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "/app/uploads"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/app/outputs"))
MAX_FILE_SIZE = int(os.getenv("OCR_MAX_FILE_SIZE", "10485760"))  # 10MB
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

# 创建必要的目录
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 分辨率配置映射
RESOLUTION_CONFIGS = {
    "tiny": {"base_size": 512, "image_size": 512, "crop_mode": False},
    "small": {"base_size": 640, "image_size": 640, "crop_mode": False},
    "base": {"base_size": 1024, "image_size": 1024, "crop_mode": False},
    "large": {"base_size": 1280, "image_size": 1280, "crop_mode": False},
    "gundam": {"base_size": 1024, "image_size": 640, "crop_mode": True},
}

# 任务类型提示词映射
TASK_PROMPTS = {
    "free_ocr": "<image>\nFree OCR.",
    "markdown": "<image>\n<|grounding|>Convert the document to markdown.",
    "parse_chart": "<image>\nParse the figure.",
    "locate_object": "<image>\nLocate <|ref|>{reference_text}<|/ref|> in the image.",
}

# 模型预热配置
WARMUP_ENABLED = os.getenv("OCR_WARMUP_ENABLED", "false").lower() == "true"
WARMUP_IMAGE_PATH = os.getenv("OCR_WARMUP_IMAGE_PATH", None)

