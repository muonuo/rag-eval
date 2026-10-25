"""答题卡：调 Agent 逐题作答，收集 (问题, 回答, 检索上下文, 标准答案) → ragas.Dataset。

retrieved_contexts 来自 agent 的偷拍器（contexts_store），
reference 是人工整理的标准答案（Context Recall 的评分尺子）。
"""

import json
import time

import config  # noqa: F401  # 统一最先 import config 的约定

from ragas import Dataset

from config import EXPERIMENTS_DIR, REQUEST_DELAY, RESOURCES_DIR

QUESTIONS_PATH = RESOURCES_DIR / "ragas_eval_dataset.json"
DATASET_NAME = "rag_eval"
_DATASET_CSV = EXPERIMENTS_DIR / "datasets" / f"{DATASET_NAME}.csv"


def load_questions(limit: int | None = None) -> list[dict]:
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = json.load(f)
    return questions[:limit] if limit else questions


def load_cached_dataset() -> Dataset | None:
    """有缓存答题卡直接复用（跑判卷不用重新答题）；没有返回 None。"""
    if not _DATASET_CSV.exists():
        return None
    dataset = _open_dataset()
    dataset.reload()
    return dataset if len(dataset) > 0 else None


def _open_dataset() -> Dataset:
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    return Dataset(name=DATASET_NAME, backend="local/csv", root_dir=str(EXPERIMENTS_DIR))


def build_dataset(
    agent,
    contexts_store: dict,
    limit: int | None = None,
    rebuild: bool = False,
) -> Dataset:
    """生成答题卡。已有缓存则复用；rebuild=True 强制重答。"""
    if rebuild:
        _DATASET_CSV.unlink(missing_ok=True)
        contexts_store.clear()

    dataset = _open_dataset()
    if _DATASET_CSV.exists():
        dataset.reload()
    if len(dataset) > 0:
        print(
            f"加载了 {len(dataset)} 条缓存答题卡，跳过重新答题"
            "（想重答加 --rebuild）"
        )
        return dataset

    eval_data = load_questions(limit)
    print(f"开始收集回答和上下文（共 {len(eval_data)} 题，间隔 {REQUEST_DELAY}s）...")
    print("-" * 50)

    for i, item in enumerate(eval_data):
        query = item["question"]
        response = agent.invoke(
            {"messages": [{"role": "user", "content": query}]},
            {"configurable": {"thread_id": f"eval-{i}"}},
        )
        dataset.append(
            {
                "user_input": query,
                "response": response["messages"][-1].content,
                "retrieved_contexts": contexts_store.get(query, []),
                "reference": item["ground_truth"],
            }
        )
        contexts = len(contexts_store.get(query, []))
        print(f"[{i + 1}/{len(eval_data)}] {query[:40]}... OK (contexts: {contexts}条)")

        if i < len(eval_data) - 1:  # 最后一条不用等
            time.sleep(REQUEST_DELAY)

    dataset.save()
    print(f"\n答题卡已保存: {len(eval_data)} 个样本 -> {_DATASET_CSV}")
    return dataset
