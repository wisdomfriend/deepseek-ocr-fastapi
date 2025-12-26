"""
API 路由注册
"""
from fastapi import APIRouter

from app.api.endpoints import health, ocr, upload

# 创建路由
api_router = APIRouter()

# 注册健康检查路由
api_router.add_api_route("/health", health.health_check, methods=["GET"], tags=["health"])
api_router.add_api_route("/", health.root, methods=["GET"], tags=["health"])

# 注册OCR路由
api_router.add_api_route("/api/ocr", ocr.upload_and_process, methods=["POST"], tags=["ocr"])
api_router.add_api_route("/api/tasks/{task_id}", ocr.get_task_status, methods=["GET"], tags=["ocr"])
api_router.add_api_route("/api/tasks", ocr.list_tasks, methods=["GET"], tags=["ocr"])
api_router.add_api_route("/binary_ocr", ocr.binary_ocr_endpoint, methods=["POST"], tags=["ocr"])
api_router.add_api_route("/upload", upload.upload_image_endpoint, methods=["POST"], tags=["upload"])
api_router.add_api_route("/upload_pdf", upload.upload_pdf_endpoint, methods=["POST"], tags=["upload"])

