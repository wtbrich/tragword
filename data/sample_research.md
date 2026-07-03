# 示例资料：向量检索与 LangGraph

向量检索常用于语义搜索。它会把文本编码成向量，再通过相似度计算找到和查询最相关的内容。

LangChain 负责统一封装 LLM、Embedding、Retriever、Tool 等能力，适合快速搭建 RAG 系统。

LangGraph 适合多步骤、可循环的工作流编排，例如 Planner → Retriever → Writer → Reviewer 的研究助手。

在实践中，研究助手通常需要：

1. 将复杂问题拆成子问题
2. 用本地知识库和联网搜索补充证据
3. 根据审稿意见多轮改写
4. 输出带引用的 Markdown 报告
