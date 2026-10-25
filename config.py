"""全局配置：路径、模型名、检索参数。

约定：所有模块第一行 import config——保证 .env 在任何模型客户端读取
环境变量之前加载完成。
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # 读取项目根 .env（DEEPSEEK_API_KEY / DASHSCOPE_API_KEY）

import compat  # noqa: F401  # ragas 兼容垫片，必须先于任何 import ragas 加载

BASE_DIR = Path(__file__).resolve().parent
RESOURCES_DIR = BASE_DIR / "resources"
DB_DIR = BASE_DIR / "db"
EXPERIMENTS_DIR = BASE_DIR / "experiments"

# 文本模型：考生答题 + 判卷 LLM 都用它。
# flash 是混合思考模型，工具调用时必须显式关思考，见 agent.py。
AGENT_MODEL = "deepseek-flash"

# 检索段：稠密向量（DashScope）+ 云端重排（与 08.Milvus 用的是同一个云模型）
EMBEDDING_MODEL = "qwen3.7-text-embedding-flash"
RERANKER_MODEL = "qwen3.7-text-rerank"

# 判卷段：LLM 走 DeepSeek，向量走本地 Ollama（OpenAI 兼容接口，api_key 随便填非空）
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
JUDGE_LLM = "deepseek-flash"
JUDGE_EMBEDDING_BASE_URL = "http://localhost:11434/v1"
JUDGE_EMBEDDING_MODEL = "qwen3-embedding:0.6b"  # 前置：ollama pull qwen3-embedding:0.6b

# 检索参数
TOP_K = 5        # 每路检索及最终进入上下文的文档数
RRF_K = 60       # RRF 平滑常数（论文经验值，一般不动）
CHUNK_SIZE = 200 # 知识库切片大小（字符）

# DeepSeek 限流保护：每次答题后的间隔（秒）
REQUEST_DELAY = 1.0
