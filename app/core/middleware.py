"""
中间件
"""
import time
import logging

logger = logging.getLogger(__name__)


async def log_requests_middleware(request, call_next):
    """记录所有请求的中间件"""
    start_time = time.time()

    # 记录请求信息
    logger.info(
        f"收到OCR请求 | {request.method} {request.url.path} | "
        f"IP: {request.client.host if request.client else 'unknown'}"
    )

    # 记录请求头
    user_agent = request.headers.get("user-agent", "unknown")
    content_type = request.headers.get("content-type", "unknown")
    logger.debug(
        f"请求头 | User-Agent: {user_agent[:50]}... | Content-Type: {content_type}"
    )

    # 处理请求
    response = await call_next(request)

    # 计算处理时间
    process_time = time.time() - start_time

    # 记录响应信息
    logger.info(
        f"OCR响应完成 | {request.method} {request.url.path} | "
        f"状态码: {response.status_code} | 耗时: {process_time:.3f}s"
    )

    # 添加处理时间到响应头
    response.headers["X-Process-Time"] = str(process_time)

    return response

