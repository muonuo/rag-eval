"""Agentic RAG：检索封装成工具，模型自主决定何时查库。

contexts_store 偷偷记录每次检索的原文——RAGAS 要分别评"检索质量"和
"生成质量"，光看最终回答推不出 Agent 实际用了哪些文档，必须偷拍中间过程。
"""

from langchain.agents import AgentState, create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import ToolRuntime, tool

from config import AGENT_MODEL, TOP_K

SYSTEM_PROMPT = """
你是一个教资考试答题助手。你必须严格遵守以下规则：

1. 有关教资的问题**只能**调用`search_knowledge_base`工具来查阅相关文档，根据文档信息回答问题
2. **禁止**添加任何你自己的知识、理解或推断
3. **禁止**扩展、举例或给出建议
4. 如果参考文档中没有相关信息，直接回复"根据提供的资料，无法回答此问题"
5. 回答要简洁，直接引用原文，不要改写或总结
"""


def build_agent(retriever) -> tuple:
    """返回 (agent, contexts_store)。store 按"原始提问"攒检索上下文，评估时取用。"""
    # deepseek-flash 是混合思考模型，思考模式不支持工具调用强转，必须显式关掉
    model = init_chat_model(AGENT_MODEL, extra_body={"thinking": {"type": "disabled"}})
    contexts_store: dict[str, list[str]] = {}

    @tool
    def search_knowledge_base(query: str, runtime: ToolRuntime[AgentState]) -> str:
        """搜索知识库，获取教育学、名人等的知识。需要查找资料时使用。"""
        final_docs = retriever.search(query, TOP_K)
        if not final_docs:
            return "未找到相关文档"

        # 顺手存 context：一次检索两用（喂给模型 + 留给评估），避免重复检索
        original_question = runtime.state["messages"][0].content
        existing = contexts_store.get(original_question, [])
        contexts_store[original_question] = list(
            {*(d["content"] for d in final_docs), *existing}
        )

        return "\n\n".join(d["content"] for d in final_docs)

    agent = create_agent(
        model=model,
        tools=[search_knowledge_base],
        system_prompt=SYSTEM_PROMPT,
    )
    return agent, contexts_store
