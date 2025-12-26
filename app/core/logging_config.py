"""
日志配置模块
"""
import os
import logging


def setup_logging():
    """配置日志"""
    LOG_LEVEL = os.getenv("OCR_LOG_LEVEL", "INFO")
    LOG_FORMAT = os.getenv(
        "OCR_LOG_FORMAT",
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # 如果还没有配置日志，则进行配置
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=getattr(logging, LOG_LEVEL.upper()),
            format=LOG_FORMAT
        )

