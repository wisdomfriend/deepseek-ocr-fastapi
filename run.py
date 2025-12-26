#!/usr/bin/env python3
"""
DeepSeek OCR Service Startup Script - For Development and Testing Environment
用于开发环境和测试环境启动OCR服务
支持热重载和调试功能
"""
import os
import logging
from app.main import app
from app.core.logging_config import setup_logging
import uvicorn

# 配置日志
setup_logging()
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    HOST = os.getenv("OCR_HOST", "0.0.0.0")
    PORT = int(os.getenv("OCR_PORT", "8000"))
    RELOAD = os.getenv("OCR_RELOAD", "false").lower() == "true"
    TIMEOUT_KEEP_ALIVE = int(os.getenv("OCR_TIMEOUT_KEEP_ALIVE", "300"))
    LOG_LEVEL = os.getenv("OCR_LOG_LEVEL", "INFO")
    
    logger.info("=" * 50)
    logger.info("Starting DeepSeek OCR Service in Development/Testing Mode")
    logger.info(f"Host: {HOST}")
    logger.info(f"Port: {PORT}")
    logger.info(f"Log Level: {LOG_LEVEL}")
    logger.info(f"Reload (Hot Reload): {RELOAD}")
    logger.info(f"Timeout Keep Alive: {TIMEOUT_KEEP_ALIVE}s")
    logger.info("Note: Use uvicorn for development (supports hot reload and debugging)")
    logger.info("Note: Workers not used in development mode (single process for debugging)")
    logger.info("=" * 50)
    

    uvicorn.run(
        "app.main:app",
        host=HOST,
        port=PORT,
        reload=RELOAD,  # Hot reload for development
        loop="asyncio",
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE,  # Keep-alive timeout
        log_level=LOG_LEVEL.lower(),
        access_log=True,
    )

