#!/bin/bash
# Docker entrypoint script - allows configurable port

# Get port from environment variable, default to 8000
OCR_PORT=${OCR_PORT:-8000}

# Get worker count from environment variable, default to 1
# Note: For DeepSeek OCR with vLLM, we typically use 1 worker due to GPU memory constraints
OCR_WORKERS=${OCR_WORKERS:-1}

# Start gunicorn with configurable port
exec gunicorn \
    --bind "0.0.0.0:${OCR_PORT}" \
    --workers ${OCR_WORKERS} \
    --worker-class uvicorn.workers.UvicornWorker \
    --timeout 300 \
    --keep-alive 5 \
    --max-requests 1000 \
    --max-requests-jitter 100 \
    --access-logfile - \
    --error-logfile - \
    --log-level info \
    app.main:app

