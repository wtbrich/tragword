from __future__ import annotations

import argparse
from pathlib import Path

from app.rag.chunking import chunk_documents
from app.rag.store import add_documents, load_documents_from_directory


def ingest_directory(data_dir: str | Path) -> int:
    documents = load_documents_from_directory(data_dir)
    chunks = chunk_documents(documents)
    return add_documents(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest local documents into Milvus.")
    parser.add_argument("--data-dir", default="data", help="资料目录，默认 data/")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise SystemExit(f"目录不存在：{data_dir}")

    documents = load_documents_from_directory(data_dir)
    if not documents:
        print(f"未找到可入库文档：{data_dir}")
        return

    chunks = chunk_documents(documents)
    count = add_documents(chunks)
    print(f"已入库 {count} 个切块，来源文档 {len(documents)} 个。")


if __name__ == "__main__":
    main()
