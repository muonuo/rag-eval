"""环境兼容垫片（源自课程 notebook 3.4"环境兼容性修复"格）。

新版 langchain-community 移除了 vertexai 聊天模块，ragas 0.4.3 却仍然
无条件 import 它。在任何 import ragas 之前注册一个空壳模块顶替，
ragas 才能正常导入。由 config.py 最先加载，其他模块无需关心。
"""

import sys
import types

import langchain_core.language_models as _lcm

_mod = types.ModuleType("langchain_community.chat_models.vertexai")


class ChatVertexAI(_lcm.BaseChatModel):
    pass


_mod.ChatVertexAI = ChatVertexAI
sys.modules["langchain_community.chat_models.vertexai"] = _mod
