from __future__ import annotations

import json
from urllib.error import URLError
from urllib.request import Request, urlopen

import streamlit as st


def call_api(base_url: str, topic: str) -> dict:
    payload = json.dumps({"topic": topic}).encode("utf-8")
    request = Request(
        f"{base_url.rstrip('/')}/research",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


st.set_page_config(page_title="Tragword Research Assistant", layout="wide")
st.title("Tragword Multi-Agent Research Assistant")

base_url = st.text_input("API Base URL", value="http://127.0.0.1:8000")
topic = st.text_area("研究题目", value="构建一个面向企业知识库的 Multi-Agent 研究助手")

if st.button("开始研究"):
    try:
        with st.spinner("研究中..."):
            result = call_api(base_url, topic)
        st.success("完成")
        st.markdown(result["final_report"])
    except URLError as exc:
        st.error(f"无法连接 API：{exc}")
    except Exception as exc:
        st.error(f"执行失败：{exc}")
