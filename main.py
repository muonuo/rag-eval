"""CLI 入口。

用法：
  uv run python main.py build [--limit N] [--rebuild]     # 生成答题卡
  uv run python main.py evaluate [--test-size 0.3] [--metrics fast|full] [--name xxx]
  uv run python main.py sample [--index 3]                # 单题跑全 6 指标（冒烟用）
"""

import argparse
import asyncio
import sys

# Windows GBK 控制台编不了部分字符（✓ 之类），强制 stdout 用 UTF-8 防崩溃
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import config  # noqa: F401  # 约定：最先 import config，保证 .env 先加载

from agent import build_agent
from dataset import build_dataset, load_cached_dataset
from evaluate import build_judges, run_experiment, score_single
from retrieval import HybridRetriever, load_knowledge_base


def make_pipeline():
    """知识库 → 混合检索 → Agent。返回 (agent, contexts_store)。"""
    docs = load_knowledge_base()
    retriever = HybridRetriever(docs)
    return build_agent(retriever)


def cmd_build(args) -> None:
    agent, contexts_store = make_pipeline()
    build_dataset(agent, contexts_store, limit=args.limit, rebuild=args.rebuild)


def cmd_evaluate(args) -> None:
    # 优先复用缓存答题卡——跑判卷不用重新加载检索管线和答题
    dataset = load_cached_dataset()
    if dataset is None:
        print("没有缓存答题卡，先现场答题生成...")
        agent, contexts_store = make_pipeline()
        dataset = build_dataset(agent, contexts_store, limit=args.limit)

    build_judges()
    asyncio.run(
        run_experiment(
            dataset,
            test_size=args.test_size,
            metrics=args.metrics,
            name=args.name,
        )
    )


def cmd_sample(args) -> None:
    dataset = load_cached_dataset()
    if dataset is None:
        raise SystemExit("没有缓存答题卡，请先运行: uv run python main.py build")

    build_judges()
    asyncio.run(score_single(dataset, index=args.index))


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 系统评估（RAGAS）")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="生成答题卡（调 Agent 答题）")
    p_build.add_argument("--limit", type=int, default=None, help="只答前 N 题")
    p_build.add_argument("--rebuild", action="store_true", help="丢弃缓存强制重答")
    p_build.set_defaults(func=cmd_build)

    p_eval = sub.add_parser("evaluate", help="批量判卷出成绩单")
    p_eval.add_argument("--limit", type=int, default=None, help="（无缓存时）只答前 N 题")
    p_eval.add_argument("--test-size", type=float, default=0.3,
                        help="判卷抽样比例（默认 0.3，notebook 原版是 0.1 只考 2 题）")
    p_eval.add_argument("--metrics", choices=["fast", "full"], default="fast",
                        help="fast=2 核心指标；full=全 6 指标（更慢）")
    p_eval.add_argument("--name", default="rag_baseline", help="实验名（成绩单目录名）")
    p_eval.set_defaults(func=cmd_evaluate)

    p_sample = sub.add_parser("sample", help="单题跑全 6 指标（批量前冒烟）")
    p_sample.add_argument("--index", type=int, default=3, help="答题卡行号（默认 3）")
    p_sample.set_defaults(func=cmd_sample)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
