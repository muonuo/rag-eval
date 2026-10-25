"""检索管线：知识库切分 → 稠密(DashScope) + 稀疏(BM25/jieba) → RRF 融合 → 阿里云重排。

对应 notebook 3.4 的 cell-15/17。课程原用 pkuseg 分词（需 VS C++ 现场编译），
此处为纯 Python 的 jieba 等价替代；本地 Cross-Encoder 重排换成阿里云
qwen3.7-text-rerank（与 08.Milvus 的云端重排是同一个模型），从而不再依赖
torch / sentence-transformers。
"""

import os

import config  # noqa: F401  # 约定：最先 import config，保证 .env 先加载

import bm25s
import jieba
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from dashscope import TextReRank

from config import CHUNK_SIZE, DB_DIR, EMBEDDING_MODEL, RERANKER_MODEL, RESOURCES_DIR, RRF_K, TOP_K

_KB_PATH = RESOURCES_DIR / "评估.md"
_BM25_INDEX_PATH = DB_DIR / "rag_eval.bm25"


# ---------- 1. 知识库加载与切分 ----------

def load_knowledge_base() -> list[Document]:
    """按 Markdown 三级标题切分；超过 CHUNK_SIZE 的节再按段落/句子递归细切。

    每个切片拼上自己的三级标题并编号（doc_1, doc_2...），
    检索结果可读，且 RRF 能按 id 跨两路去重。
    """
    with open(_KB_PATH, encoding="utf-8") as f:
        raw = f.read()

    md_chunks = MarkdownHeaderTextSplitter(
        [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
    ).split_text(raw)
    recursive = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=0, separators=["\n\n", "\n", "。"]
    )

    docs: list[Document] = []
    for chunk in md_chunks:
        pieces = (
            recursive.split_documents([chunk])
            if len(chunk.page_content) > CHUNK_SIZE
            else [chunk]
        )
        for piece in pieces:
            header = piece.metadata.get("Header 3", "")
            if header:
                piece.page_content = f"### {header}\n{piece.page_content}"
            docs.append(piece)

    for i, doc in enumerate(docs, start=1):
        doc.id = f"doc_{i}"

    print(f"知识库已就绪: {len(docs)} 个文档切片")
    return docs


# ---------- 2. 稠密检索（DashScope 向量，双塔可预计算所以快）----------

def _build_vectorstore(docs: list[Document]) -> InMemoryVectorStore:
    embeddings = DashScopeEmbeddings(
        model=EMBEDDING_MODEL,
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
    )
    vectorstore = InMemoryVectorStore(embeddings)
    vectorstore.add_documents(docs)
    return vectorstore


# ---------- 3. 稀疏检索（BM25 认字不懂意思，和稠密互补盲区）----------

def _build_bm25(docs: list[Document]) -> bm25s.BM25:
    """jieba 分词建索引，缓存到 ./db/，二次启动免重建。"""
    corpus = [{"id": doc.id, "content": doc.page_content} for doc in docs]
    DB_DIR.mkdir(exist_ok=True)

    if _BM25_INDEX_PATH.exists():
        retriever = bm25s.BM25.load(str(_BM25_INDEX_PATH), load_corpus=True)
        print(f"BM25 索引已从缓存加载: {_BM25_INDEX_PATH}")
        return retriever

    tokens = [jieba.lcut(doc["content"]) for doc in corpus]
    retriever = bm25s.BM25(k1=1.5, b=0.75, corpus=corpus)
    retriever.index(tokens)
    retriever.save(str(_BM25_INDEX_PATH))
    print(f"BM25 索引已创建并保存: {_BM25_INDEX_PATH}")
    return retriever


def _bm25_search(retriever: bm25s.BM25, query: str, k: int) -> list[dict]:
    results, _scores = retriever.retrieve([jieba.lcut(query)], k=k)
    return [results[0, i] for i in range(results.shape[1])]


# ---------- 4. RRF 融合（名次投票，不比分数）----------

def reciprocal_rank_fusion(ranked_lists: list[list[dict]], k: int = RRF_K) -> list[dict]:
    """每路按名次贡献 1/(k+rank) 并累加。

    两路分数量纲不可比（余弦 0~1 vs BM25 词频分），但名次天然可比——
    等价于两个评委投票：两路都靠前的文档必然胜过只有一路说好的。
    """
    scores: dict[str, float] = {}
    docs: dict[str, dict] = {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            scores[doc["id"]] = scores.get(doc["id"], 0.0) + 1.0 / (k + rank)
            docs[doc["id"]] = doc

    ranked_ids = sorted(scores, key=scores.get, reverse=True)
    return [docs[i] for i in ranked_ids]


# ---------- 5. 云端重排（逐对精算相关度，只精排小候选）----------

def _cloud_rerank(query: str, docs: list[dict], top_k: int) -> list[dict]:
    """阿里云 qwen3.7-text-rerank 精排（v1 接口；dashscope 2.0 的 client.rerank 端点目前 404）。

    relevance_score 为 0~1 分制，== 0 视为不相关丢弃——宁可少给、不给错的，
    直接利好 Faithfulness。
    """
    if not docs:
        return []
    resp = TextReRank.call(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model=RERANKER_MODEL,
        query=query,
        documents=[d["content"] for d in docs],
        top_n=top_k,
        return_documents=False,
    )
    if not resp.output or not resp.output.results:
        return []
    ranked = sorted(
        resp.output.results, key=lambda r: r["relevance_score"], reverse=True
    )
    return [
        {**docs[r["index"]], "score": r["relevance_score"]}
        for r in ranked
        if r["relevance_score"] > 0
    ]


# ---------- 完整管线 ----------

class HybridRetriever:
    """稠密 + BM25 + RRF + 云端重排的混合检索管线。"""

    def __init__(self, docs: list[Document]):
        self.vectorstore = _build_vectorstore(docs)
        self.bm25 = _build_bm25(docs)
        print("所有检索组件准备完成！（稠密 + BM25 + RRF + 云端重排）")

    def search(self, query: str, top_k: int = TOP_K) -> list[dict]:
        dense_docs = self.vectorstore.similarity_search(query, k=top_k)
        sparse_docs = _bm25_search(self.bm25, query, k=top_k)
        if not dense_docs or not sparse_docs:
            return []

        dense_rs = [{"id": d.id, "content": d.page_content} for d in dense_docs]
        fused = reciprocal_rank_fusion([dense_rs, sparse_docs])
        return _cloud_rerank(query, fused, top_k)
