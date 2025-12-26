"""
FastAPI 应用主入口
"""
import logging
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.core.logging_config import setup_logging
from app.core.lifespan import lifespan
from app.core.middleware import log_requests_middleware
from app.core.exceptions import (
    not_found_handler,
    global_exception_handler,
    validation_exception_handler
)
from app.core.config import OUTPUT_DIR
from app.api.routes import api_router

# 配置日志
setup_logging()
logger = logging.getLogger(__name__)

# 创建 FastAPI 应用
app = FastAPI(
    title="DeepSeek OCR API Service",
    description="High-performance OCR service based on DeepSeek-OCR",
    version="1.0.0",
    lifespan=lifespan,
    root_path="/deepseek-ocr"
)

# CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册中间件
app.middleware("http")(log_requests_middleware)

# 注册异常处理器
app.add_exception_handler(404, not_found_handler)
app.add_exception_handler(Exception, global_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)

# 注册路由
app.include_router(api_router)

# 静态文件服务
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

