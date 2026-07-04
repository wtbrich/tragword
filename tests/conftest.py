from app.rag.store import reset_vectorstore
from pytest import fixture


@fixture(autouse=True)
def _reset_milvus_connections():
    reset_vectorstore()
    yield
    reset_vectorstore()
