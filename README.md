# rag-eval：RAG 系统评估（RAGAS 六指标）

源自黑马 LangChain 课程 3.4《RAG 评估》实践，从 notebook 抽出的独立可复用项目。
2026-11 混合架构简历项目的评估模块可直接参考本仓库。

## 流程全景

```
评估.md 知识库（30 切片）
   │
问题 ─→ ①双塔稠密(DashScope qwen3.7-text-embedding-flash) ─┐
        ②BM25稀疏(jieba分词)                             ─┴→ ③RRF名次投票
   │                                                            ↓
   │                    ④阿里云重排 qwen3.7-text-rerank（0 分丢弃，取 top-5）
   ↓
⑤Agent(deepseek-flash) 拿文档生成回答
   │  （工具顺手把检索原文存进 contexts_store，供评估偷拍）
   ↓
答题卡：14 题 × (user_input / response / retrieved_contexts / reference)
   ↓
裁判：DeepSeek(deepseek-flash) 判卷 + 本地 Ollama(qwen3-embedding:0.6b) 算向量
   ↓
成绩单：RAGAS 六指标 → experiments/experiments/<实验名>/
```

## 六指标速查

| 考谁 | 指标 | 一句话 | 方向 |
|---|---|---|---|
| 检索器 | context_precision | 捞回来的准不准、排序对不对 | 越高越好 |
| 检索器 | context_recall | 标准答案要点覆盖了没 | 越高越好 |
| 检索器 | context_entity_recall | 人名/书名等实体漏没漏 | 越高越好 |
| 检索器 | noise_sensitivity | 答案有没有被噪声文档带偏 | **越低越好** |
| 生成器 | faithfulness | 有没有编造（反幻觉） | 越高越好 |
| 生成器 | answer_relevancy | 答得切题不切题 | 越高越好 |

## 快速开始

```bash
# 0. 前置
#    - Ollama 已启动，且已拉判卷向量模型（一次性）：
ollama pull qwen3-embedding:0.6b
#    - .env 里有 DEEPSEEK_API_KEY / DASHSCOPE_API_KEY

# 1. 安装依赖（轻量，无 torch）
uv sync

# 2. 配置密钥
cp .env.example .env   # 填入两个 key

# 3. 生成答题卡（14 题约 5 分钟；--limit 2 可先冒烟）
uv run python main.py build

# 4. 单题跑全 6 指标（批量前确认裁判都能出分）
uv run python main.py sample --index 3

# 5. 批量判卷出成绩单
uv run python main.py evaluate --test-size 0.3 --metrics fast
uv run python main.py evaluate --test-size 0.3 --metrics full   # 全 6 指标，更慢
```

## 相对 notebook 的改动

1. `test_size` 默认 0.1 → **0.3**（原版只考 2 题，运气成分大）
2. 新增 `--metrics fast|full`：批量实验支持全 6 指标（原版只 2 个）
3. **重排从本地 CrossEncoder(Qwen3-Reranker-0.6B) 换成阿里云 qwen3.7-text-rerank**
   （与 08.Milvus 云端重排同一模型；去掉 torch / sentence-transformers 两个大依赖）
4. 文本模型统一 **deepseek-flash**（答题 + 判卷）；判卷向量走本地 Ollama qwen3-embedding
5. 路径锚定项目根，摆脱 cwd 相对路径；去掉逐切片调试打印
6. 判卷抽样子集存为独立数据集 `rag_eval_test`（原版会覆盖全量答题卡缓存）
7. `build / evaluate / sample` 三个子命令 + `--rebuild` 强制重答

## 目录结构

```
├── config.py       # 路径/模型名/检索参数（改模型只动这里）
├── retrieval.py    # 知识库切分 + 稠密 + BM25 + RRF + 云端重排
├── agent.py        # init_chat_model(关思考) + 检索工具 + contexts_store 偷拍器
├── dataset.py      # 答题卡生成/缓存（ragas Dataset, local/csv）
├── evaluate.py     # 双裁判 + 六指标 + @experiment 批量
├── main.py         # CLI 入口
├── resources/      # 知识库 评估.md + 14 题 QA 对
├── db/             # BM25 索引缓存（首跑自动生成）
└── experiments/    # 答题卡 datasets/ + 成绩单 experiments/
```

## 换一个知识库/题库

把 `resources/评估.md` 换成你的知识库（保持 #/##/### 三级标题结构）、
`resources/ragas_eval_dataset.json` 换成 `[{question, ground_truth}, ...]`，
删掉 `db/`、`experiments/` 后重跑即可。标准答案请从知识库原文提炼，
否则 context_recall 会被标准答案"冤枉"。

## 已知坑

- **判卷 NaN**：deepseek-flash 是混合思考模型，思考内容可能干扰判卷 JSON 解析。
  批量出现 NaN 时把 `config.JUDGE_LLM` 换成 `deepseek-chat`
- **限流**：答题间隔 `config.REQUEST_DELAY`（默认 1s），报限流错误就调大
