FROM image.sourcefind.cn:5000/dcu/admin/base/vllm:0.8.5-ubuntu22.04-dtk25.04.1-rc5-das1.6-py3.10-20250724

# 设置工作目录
WORKDIR /app

# 复制DeepSeek-OCR-vllm目录
COPY DeepSeek-OCR-vllm /app/DeepSeek-OCR-vllm

# 复制应用代码
COPY app/ /app/app/
COPY run.py /app/
COPY requirements.txt /app/

# 安装额外的Python依赖
RUN pip install --no-cache-dir -r requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --trusted-host pypi.tuna.tsinghua.edu.cn

# 创建必要的目录
RUN mkdir -p /app/uploads /app/outputs

# 复制 entrypoint 脚本
COPY docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

# 设置环境变量
ENV PYTHONPATH=/app
ENV VLLM_USE_V1=0
ENV CUDA_VISIBLE_DEVICES=0

# 暴露端口（可通过环境变量覆盖）
ARG OCR_PORT=8000
EXPOSE ${OCR_PORT}

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD sh -c 'OCR_PORT=${OCR_PORT:-8000}; python -c "import requests; requests.get(\"http://localhost:${OCR_PORT}/health\", timeout=5)"' || exit 1

# 使用 entrypoint 脚本启动 gunicorn（生产环境）
ENTRYPOINT ["/app/docker-entrypoint.sh"]
