# Tragword Multi-Agent Research Assistant

这是一个学习向的 **LangGraph + LangChain** 多智能体研究助手脚手架。

流程：

1. 输入研究题目
2. Planner 拆解子问题
3. Retriever 使用本地向量检索 + 可选 DDGS 联网搜索收集资料
4. Writer 基于资料生成 Markdown 草稿
5. Reviewer 审核
6. 不合格则回炉重写，直到通过或达到最大修订次数

## 目录说明

- `app/config.py`：环境变量配置
- `app/llm.py`：OpenAI 兼容 LLM 工厂
- `app/rag/`：切块、Embedding、Milvus 存储、混合检索、重排
- `app/agents/`：Planner / Retriever / Writer / Reviewer
- `app/graph/`：LangGraph 状态、编排与流式事件
- `app/api.py`：FastAPI `POST /research` 和 `POST /research/stream`
- `scripts/ingest.py`：把 `data/` 里的文档入库
- `ui/streamlit_app.py`：可选前端，支持实时流式输出

## 快速开始

### 1. 创建虚拟环境并安装依赖

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env`，按需修改：

```bash
cp .env.example .env
```

### 3. 入库示例资料

```bash
python scripts/ingest.py --data-dir data
```

### 4. 启动 API

```bash
uvicorn app.api:app --reload
```

### 5. 调用研究接口

```bash
curl -X POST http://127.0.0.1:8000/research \
  -H "Content-Type: application/json" \
  -d '{"topic":"构建一个面向企业知识库的 Multi-Agent 研究助手"}'
```

流式接口：

```bash
curl -N -X POST http://127.0.0.1:8000/research/stream \
  -H "Content-Type: application/json" \
  -d '{"topic":"构建一个面向企业知识库的 Multi-Agent 研究助手"}'
```

## 环境变量

- `OPENAI_API_KEY`：OpenAI 兼容接口 Key
- `OPENAI_BASE_URL`：兼容接口地址，可空
- `LLM_MODEL`：默认 `gpt-4o-mini`
- `EMBEDDING_PROVIDER`：`huggingface` 或 `openai`
- `EMBEDDING_MODEL`：默认 `BAAI/bge-small-zh-v1.5`
- `MILVUS_DB_URI`：默认 `./.milvus/milvus_local.db`（不用 `MILVUS_URI`，该名与 pymilvus 保留环境变量冲突）
- `MILVUS_COLLECTION_NAME`：默认 `tragword`
- `HYBRID_ENABLED`：是否启用 Milvus + BM25 混合检索
- `RERANK_ENABLED`：是否启用 cross-encoder 重排
- `RERANK_MODEL`：默认 `BAAI/bge-reranker-base`
- `RERANK_CANDIDATES`：重排前候选池大小
- `MULTIQUERY_ENABLED`：是否启用多查询扩展，默认关闭
- `SEARCH_ENABLED`：是否启用 DDGS 搜索
- `MAX_REVISIONS`：最大回炉次数，防止死循环

## Milvus Lite 本地模式

默认使用 **Milvus Lite** 的本地 `.db` 文件模式：

```bash
MILVUS_DB_URI=./.milvus/milvus_local.db
```

如果后续想切换到远程 Milvus / Zilliz Cloud，只需要改 `MILVUS_DB_URI`，例如：

```bash
MILVUS_DB_URI=http://127.0.0.1:19530
```

同一个 `MILVUS_COLLECTION_NAME` 下即可复用现有入库逻辑。

## 流式 UI

- 本项目默认使用本地 `bge-small-zh-v1.5` 做 Embedding，适合离线检索。
- LLM 通过 `ChatOpenAI` 封装，支持 OpenAI / DeepSeek / 通义等 OpenAI 兼容服务。
- 如果没有配置 Key，代码依然可以 import 和启动；真正执行研究任务时会尽量优雅降级。
- Streamlit 前端提供“实时流式输出”开关，会订阅 `/research/stream` 并逐步展示 planner / retrieve_one / writer / reviewer 事件。
