import asyncio
import time
from fastapi import FastAPI
from contextlib import asynccontextmanager
import uvicorn
from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.model_executor.models.registry import ModelRegistry
from deepseek_ocr import DeepseekOCRForCausalLM

# 1. 模型路径配置
MODEL_PATH = "/models/DeepSeek-OCR"

# 2. 注册自定义OCR模型到vllm
ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)

# 3. FastAPI生命周期管理：模型加载与保活
@asynccontextmanager
async def lifespan(app: FastAPI):
    """在FastAPI启动时加载模型，关闭时释放资源"""
    # 启动阶段：加载模型并绑定到app.state（全局状态）
    print(f"正在加载模型：{MODEL_PATH}...")
    start_time = time.time()
    
    # 配置vllm引擎参数
    engine_args = AsyncEngineArgs(
        model=MODEL_PATH,
        hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
        block_size=64,
        max_model_len=8192,
        enforce_eager=False,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.95,
    )
    
    # 初始化模型引擎（核心：加载模型到GPU）
    app.state.engine = AsyncLLMEngine.from_engine_args(engine_args)
    
    # 打印加载完成信息
    load_duration = time.time() - start_time
    print(f"✅ 模型启动完成！加载耗时：{load_duration:.2f}秒")
    print("ℹ️  模型已就绪，绑定到FastAPI应用实例，持续运行中...")
    
    yield  # 服务运行中：模型保持加载状态
    
    # 关闭阶段：释放模型资源（可选）
    if hasattr(app.state, "engine"):
        del app.state.engine
        print("模型资源已释放")

# 4. 创建FastAPI应用实例
app = FastAPI(
    title="DeepSeek-OCR 模型服务",
    description="通过FastAPI+Uvicorn保持模型持续启动状态",
    lifespan=lifespan  # 绑定生命周期管理
)

# 5. 可选：添加健康检查接口（验证模型是否存活）
@app.get("/health")
async def health_check():
    """检查模型服务是否正常运行"""
    if hasattr(app.state, "engine"):
        return {
            "status": "healthy",
            "model_path": MODEL_PATH,
            "message": "模型持续运行中，可接收推理请求"
        }
    else:
        return {
            "status": "unhealthy",
            "message": "模型未加载"
        }

# 6. 启动Uvicorn服务（保持进程运行）
if __name__ == "__main__":
    uvicorn.run(
        "run:app",  # 指向当前文件的FastAPI实例
        host="0.0.0.0",  # 允许外部访问
        port=8000,       # 服务端口
        workers=1,       # 单进程（避免模型重复加载）
        timeout_keep_alive=300,  # 长连接超时设置
        loop="asyncio"   # 使用异步事件循环
    )
