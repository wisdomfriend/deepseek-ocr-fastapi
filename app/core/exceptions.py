"""
异常处理
"""
import logging
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

logger = logging.getLogger(__name__)


async def not_found_handler(request: Request, exc):
    """处理 404 错误"""
    logger.warning(
        f"404 错误 | {request.method} {request.url.path} | "
        f"IP: {request.client.host if request.client else 'unknown'}"
    )
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": f"Path '{request.url.path}' not found",
            "available_endpoints": [
                "/",
                "/health",
                "/api/ocr",
                "/api/tasks",
                "/upload",
                "/upload_pdf",
                "/binary_ocr"
            ]
        }
    )


async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理器"""
    logger.error(
        f"未处理异常 | {request.method} {request.url.path} | "
        f"IP: {request.client.host if request.client else 'unknown'} | "
        f"错误: {str(exc)}",
        exc_info=True
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred during OCR processing",
            "path": request.url.path
        }
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """请求验证异常处理器"""
    logger.error(f"422 请求验证失败: {exc.errors()}")

    # 处理body，避免序列化UploadFile对象
    body_info = None
    if exc.body:
        if hasattr(exc.body, 'filename'):  # UploadFile对象
            body_info = f"UploadFile: {exc.body.filename}"
        else:
            try:
                body_info = str(exc.body)
            except Exception as e:
                body_info = "Non-serializable body"

    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": body_info}
    )

