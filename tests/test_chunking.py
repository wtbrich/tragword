from app.rag.chunking import chunk_documents
from langchain_core.documents import Document


def test_chunk_documents_splits_and_keeps_metadata() -> None:
    doc = Document(
        page_content="第一段内容。第二段内容。第三段内容。第四段内容。",
        metadata={"source": "sample.md"},
    )

    chunks = chunk_documents([doc])

    assert len(chunks) >= 1
    assert all(chunk.metadata["source"] == "sample.md" for chunk in chunks)
    assert any("第一段内容" in chunk.page_content for chunk in chunks)
