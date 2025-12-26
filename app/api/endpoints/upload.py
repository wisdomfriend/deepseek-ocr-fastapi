"""
文件上传相关端点
"""
import time
import logging
from io import BytesIO
from fastapi import UploadFile, File, Form, HTTPException

from app.models.schemas import OCRUploadResponse, OCRPDFResponse
from app.services.ocr_service import run_deepseek_on_pil
from app.core.config import RESOLUTION_CONFIGS, TASK_PROMPTS
from PIL import Image

logger = logging.getLogger(__name__)


async def upload_image_endpoint(
    file: UploadFile = File(...),
    use_det: bool = Form(True),
    use_cls: bool = Form(True),
    use_rec: bool = Form(True),
    text_score: float = Form(0.5),
    box_thresh: float = Form(0.5),
    unclip_ratio: float = Form(1.6),
    return_word_box: bool = Form(False),
    task_type: str = Form("markdown"),
    resolution: str = Form("gundam")
):
    """与 ocr_server 的 /upload 同名的图片上传接口，但使用 DeepSeek 本地推理。
    兼容接收 ocr_server 的表单参数（目前本地推理未用 det/cls/rec 等开关）。
    """
    try:
        # 检查文件类型（尽量兼容常见图片）
        content = await file.read()
        start_time = time.time()
        image = Image.open(BytesIO(content)).convert('RGB')

        text, processed_text, results = await run_deepseek_on_pil(image, task_type, resolution)
        elapsed = time.time() - start_time

        return OCRUploadResponse(
            success=True,
            results=results,
            processing_time=round(elapsed, 4),
            image_size={"width": image.size[0], "height": image.size[1]},
            text=text,
            processed_text=processed_text
        )
    except Exception as e:
        logger.exception(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=f"upload error: {str(e)}")


async def upload_pdf_endpoint(
    file: UploadFile = File(...),
    task_type: str = Form("markdown"),
    resolution: str = Form("gundam")
):
    """与 ocr_server 名称保持一致的 PDF OCR 接口，使用 DeepSeek 本地推理。
    通过 PyMuPDF 渲染每一页为图像后进行识别。
    """
    try:
        try:
            import fitz  # PyMuPDF
        except Exception:
            raise HTTPException(
                status_code=500, 
                detail="需要安装 PyMuPDF (fitz) 才能处理 PDF，请在镜像中加入 pymupdf 依赖。"
            )

        pdf_bytes = await file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        results_pages = []

        for page_index in range(len(doc)):
            page = doc[page_index]
            # 将整页渲染为位图（避免依赖提取内嵌图片能力）
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x 放大，清晰些
            img_data = pix.tobytes("png")
            pil_image = Image.open(BytesIO(img_data)).convert('RGB')

            page_start = time.time()
            text, processed_text, results = await run_deepseek_on_pil(pil_image, task_type, resolution)
            page_elapsed = time.time() - page_start

            results_pages.append({
                "page": page_index + 1,
                "index": 0,
                "result": results,
                "bbox_image": [0, 0, pil_image.size[0], pil_image.size[1]],
                "processing_time": round(page_elapsed, 4),
                "image_size": {"width": pil_image.size[0], "height": pil_image.size[1]},
                "text": text,
                "processed_text": processed_text
            })

        doc.close()
        return OCRPDFResponse(success=True, results=results_pages)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Upload PDF error: {e}")
        raise HTTPException(status_code=500, detail=f"upload_pdf error: {str(e)}")

