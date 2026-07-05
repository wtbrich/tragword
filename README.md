# Tragword Multi-Agent Research Assistant

这是一个学习向的 **LangGraph + LangChain** 多智能体研究助手脚手架。

流程：

1. 输入研究题目
2. Planner 拆解子问题
3. 可选提纲审批后再进入检索
4. Retriever 使用本地向量检索 + 可选 DDGS 联网搜索收集资料
5. Writer 基于资料生成 Markdown 草稿
6. Reviewer 审核
7. 不合格则回炉重写，直到通过或达到最大修订次数

## 目录说明

- `app/config.py`：环境变量配置
- `app/llm.py`：OpenAI 兼容 LLM 工厂
- `app/rag/`：切块、Embedding、Milvus 存储、混合检索、重排
- `app/agents/`：Planner / Retriever / Writer / Reviewer
- `app/graph/`：LangGraph 状态、编排、流式事件、提纲审批与 checkpoint
- `app/api.py`：FastAPI `POST /research`、`POST /research/stream`、`POST /research/interactive/*`
- `scripts/ingest.py`：把 `data/` 里的文档入库
- `ui/streamlit_app.py`：可选前端，支持同步、实时流式输出和提纲审批模式

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

同步接口：

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

提纲审批流程：

1. `POST /research/interactive/start`
2. 人工修改返回的 `sub_questions`
3. `POST /research/interactive/resume`

## 环境变量

- `OPENAI_API_KEY`：OpenAI 兼容接口 Key
- `OPENAI_BASE_URL`：兼容接口地址，可空
- `LLM_MODEL`：默认 `gpt-4o-mini`
- `EMBEDDING_PROVIDER`：`huggingface` 或 `openai`
- `EMBEDDING_MODEL`：默认 `BAAI/bge-small-zh-v1.5`
- `MILVUS_DB_URI`：默认 `./.milvus/milvus_local.db`
- `MILVUS_COLLECTION_NAME`：默认 `tragword`
- `HYBRID_ENABLED`：是否启用 Milvus + BM25 混合检索
- `RERANK_ENABLED`：是否启用 cross-encoder 重排
- `RERANK_MODEL`：默认 `BAAI/bge-reranker-base`
- `RERANK_CANDIDATES`：重排前候选池大小
- `MULTIQUERY_ENABLED`：是否启用多查询扩展，默认关闭
- `SEARCH_ENABLED`：是否启用 DDGS 搜索
- `MAX_REVISIONS`：最大回炉次数，防止死循环
- `CHECKPOINT_DB`：提纲审批模式的 SQLite checkpoint 路径，默认 `.langgraph/checkpoints.sqlite`

## 模型预加载

首次运行会下载两个模型：

- `BAAI/bge-small-zh-v1.5`，约 93M
- `BAAI/bge-reranker-base`，约 1.1G

默认缓存目录是 `~/.cache/huggingface`，可以通过 `HF_HOME` 改到其他磁盘。国内网络下建议设置：

```bash
HF_ENDPOINT=https://hf-mirror.com
HF_HUB_DOWNLOAD_TIMEOUT=60
```

如果想先把模型预下载好，可以运行：

```bash
python scripts/warmup.py
```

如果希望服务启动时就预加载，可以设置：

```bash
EAGER_LOAD_MODELS=true
```

如果暂时不需要重排，也可以关闭它来跳过 1.1G 的 reranker 下载：

```bash
RERANK_ENABLED=false
```

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

## 流式 UI 与提纲审批模式

- 本项目默认使用本地 `bge-small-zh-v1.5` 做 Embedding，适合离线检索。
- LLM 通过 `ChatOpenAI` 封装，支持 OpenAI / DeepSeek / 通义等 OpenAI 兼容服务。
- 如果没有配置 Key，代码依然可以 import 和启动；真正执行研究任务时会尽量优雅降级。
- Streamlit 前端提供三种模式：
  - 同步执行
  - 实时流式输出（订阅 `/research/stream`）
  - 提纲审批模式（先调 `/research/interactive/start`，再调 `/research/interactive/resume`）
