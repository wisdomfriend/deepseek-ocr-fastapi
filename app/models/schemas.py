"""
Pydantic 数据模型
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel


# OCR 请求和响应模型
class OCRRequest(BaseModel):
    """OCR 请求模型"""
    prompt: Optional[str] = None
    include_visualization: bool = True
    resolution: str = "gundam"
    task_type: str = "markdown"
    reference_text: Optional[str] = None


class OCRResponse(BaseModel):
    """OCR 响应模型"""
    task_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class TaskStatus(BaseModel):
    """任务状态模型"""
    task_id: str
    status: str  # pending, processing, completed, failed
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: float
    completed_at: Optional[float] = None


class OCRResult(BaseModel):
    """OCR 结果项模型"""
    label: str
    text: str
    confidence: float
    bbox: List[List[float]]


class OCRUploadResponse(BaseModel):
    """OCR上传响应模型"""
    success: bool
    results: List[OCRResult]
    processing_time: float
    image_size: Dict[str, int]
    text: Optional[str] = None
    processed_text: Optional[str] = None


class OCRPDFPageResult(BaseModel):
    """PDF OCR 页面结果模型"""
    page: int
    index: int
    result: List[OCRResult]
    bbox_image: List[float]
    processing_time: float
    image_size: Dict[str, int]
    text: Optional[str] = None
    processed_text: Optional[str] = None


class OCRPDFResponse(BaseModel):
    """PDF OCR 响应模型"""
    success: bool
    results: List[OCRPDFPageResult]

