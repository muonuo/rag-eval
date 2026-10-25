"""判卷：LLM 走 DeepSeek，向量走本地 Ollama，RAGAS 六指标 + @experiment 批量。

deepseek-flash 是混合思考模型；若判卷批量出现 NaN（思考内容干扰 JSON 解析），
把 config.JUDGE_LLM 换成 deepseek-chat 再试。
"""

import os

import config  # noqa: F401  # 约定：最先 import config，保证 .env 先加载

from openai import AsyncOpenAI
from pydantic import BaseModel
from ragas import Dataset, experiment
from ragas.embeddings.base import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextEntityRecall,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
    NoiseSensitivity,
)

from config import (
    DEEPSEEK_BASE_URL,
    EXPERIMENTS_DIR,
    JUDGE_EMBEDDING_BASE_URL,
    JUDGE_EMBEDDING_MODEL,
    JUDGE_LLM,
)
from dataset import DATASET_NAME

# build_judges() 赋值，@experiment 函数运行时引用
_judge_llm = None
_judge_embeddings = None


def build_judges() -> None:
    """两个裁判各干合适的活：LLM 评要"读懂内容"的指标，embedding 评要"算相似度"的指标。

    LLM 走 DeepSeek（判卷质量）；向量走本地 Ollama 的 qwen3-embedding（免费，
    OpenAI 兼容接口，api_key 随便填非空——SDK 要求非空，Ollama 不校验）。
    """
    global _judge_llm, _judge_embeddings
    llm_client = AsyncOpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"), base_url=DEEPSEEK_BASE_URL
    )
    emb_client = AsyncOpenAI(api_key="ollama", base_url=JUDGE_EMBEDDING_BASE_URL)
    _judge_llm = llm_factory(model=JUDGE_LLM, client=llm_client, temperature=0)
    _judge_embeddings = embedding_factory(
        model=JUDGE_EMBEDDING_MODEL, client=emb_client
    )


class FastResult(BaseModel):
    """fast 档：两条核心指标（检索段 + 生成段各一条）。"""

    context_recall: float
    faithfulness: float


class FullResult(FastResult):
    """full 档：全 6 指标。noise_sensitivity 越低越好，其余越高越好。"""

    context_precision: float
    context_entity_recall: float
    answer_relevancy: float
    noise_sensitivity: float


@experiment(experiment_model=FastResult)
async def run_fast(row) -> FastResult:
    user_input = row.get("user_input")
    reference = row.get("reference")
    retrieved_contexts = row.get("retrieved_contexts")
    response = row.get("response")

    cr = await ContextRecall(llm=_judge_llm).ascore(
        user_input=user_input, reference=reference, retrieved_contexts=retrieved_contexts
    )
    faith = await Faithfulness(llm=_judge_llm).ascore(
        user_input=user_input, response=response, retrieved_contexts=retrieved_contexts
    )
    return FastResult(context_recall=cr.value, faithfulness=faith.value)


@experiment(experiment_model=FullResult)
async def run_full(row) -> FullResult:
    user_input = row.get("user_input")
    reference = row.get("reference")
    retrieved_contexts = row.get("retrieved_contexts")
    response = row.get("response")

    cp = await ContextPrecision(llm=_judge_llm).ascore(
        user_input=user_input, reference=reference, retrieved_contexts=retrieved_contexts
    )
    cr = await ContextRecall(llm=_judge_llm).ascore(
        user_input=user_input, reference=reference, retrieved_contexts=retrieved_contexts
    )
    cer = await ContextEntityRecall(llm=_judge_llm).ascore(
        reference=reference, retrieved_contexts=retrieved_contexts
    )
    faith = await Faithfulness(llm=_judge_llm).ascore(
        user_input=user_input, response=response, retrieved_contexts=retrieved_contexts
    )
    ar = await AnswerRelevancy(llm=_judge_llm, embeddings=_judge_embeddings).ascore(
        user_input=user_input, response=response
    )
    ns = await NoiseSensitivity(llm=_judge_llm).ascore(
        user_input=user_input,
        response=response,
        reference=reference,
        retrieved_contexts=retrieved_contexts,
    )
    return FullResult(
        context_recall=cr.value,
        faithfulness=faith.value,
        context_precision=cp.value,
        context_entity_recall=cer.value,
        answer_relevancy=ar.value,
        noise_sensitivity=ns.value,
    )


async def run_experiment(
    dataset, test_size: float, metrics: str, name: str
):
    """抽样判卷并落盘成绩单。注意：测试子集用独立 name 保存，不覆盖全量答题卡。"""
    _train, test = dataset.train_test_split(test_size=test_size)
    print(f"抽样判卷: {len(test)}/{len(dataset)} 题（test_size={test_size}）")

    tds = Dataset.from_pandas(
        test.to_pandas(),
        name=f"{DATASET_NAME}_test",
        backend="local/csv",
        root_dir=str(EXPERIMENTS_DIR),
    )
    runner = run_fast if metrics == "fast" else run_full
    await runner.arun(tds, name=name)
    print("=" * 60)
    print(f"成绩单已保存至: {EXPERIMENTS_DIR}/experiments/{name}/")


async def score_single(dataset, index: int = 3) -> None:
    """单题跑全 6 指标并逐个打印——批量前先冒烟，确认裁判都能出分。"""
    row = dataset.__getitem__(index)
    user_input = row.get("user_input")
    reference = row.get("reference")
    retrieved_contexts = row.get("retrieved_contexts")
    response = row.get("response")

    print(f"========开始评估：input: {user_input}===========")

    cp = await ContextPrecision(llm=_judge_llm).ascore(
        user_input=user_input, reference=reference, retrieved_contexts=retrieved_contexts
    )
    print(f"context_precision:      {cp.value}（1/6）")
    cr = await ContextRecall(llm=_judge_llm).ascore(
        user_input=user_input, reference=reference, retrieved_contexts=retrieved_contexts
    )
    print(f"context_recall:         {cr.value}（2/6）")
    cer = await ContextEntityRecall(llm=_judge_llm).ascore(
        reference=reference, retrieved_contexts=retrieved_contexts
    )
    print(f"context_entity_recall:  {cer.value}（3/6）")
    faith = await Faithfulness(llm=_judge_llm).ascore(
        user_input=user_input, response=response, retrieved_contexts=retrieved_contexts
    )
    print(f"faithfulness:           {faith.value}（4/6）")
    ar = await AnswerRelevancy(llm=_judge_llm, embeddings=_judge_embeddings).ascore(
        user_input=user_input, response=response
    )
    print(f"answer_relevancy:       {ar.value}（5/6）")
    ns = await NoiseSensitivity(llm=_judge_llm).ascore(
        user_input=user_input,
        response=response,
        reference=reference,
        retrieved_contexts=retrieved_contexts,
    )
    print(f"noise_sensitivity:      {ns.value}（6/6，越低越好）")
    print("========================评估完成==========================")
