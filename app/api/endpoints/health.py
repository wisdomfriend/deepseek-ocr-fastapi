"""
健康检查端点
"""
import logging
from fastapi import HTTPException
from fastapi.responses import HTMLResponse

from app.core.lifespan import get_engine
from app.utils.templates import load_template

logger = logging.getLogger(__name__)


async def health_check():
    """健康检查端点"""
    engine = get_engine()
    if engine is None:
        raise HTTPException(503, "OCR model not initialized")
    
    return {
        "status": "healthy",
        "model": "DeepSeek-OCR",
        "version": "1.0.0",
    }


async def root():
    """返回主页"""
    try:
        html_content = load_template("index.html")
        return HTMLResponse(content=html_content)
    except FileNotFoundError as e:
        logger.error(f"Template not found: {e}")
        raise HTTPException(status_code=500, detail="Template file not found")

