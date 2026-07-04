from __future__ import annotations

import json
from urllib.error import URLError
from urllib.request import Request, urlopen

import streamlit as st


def _call_json_api(base_url: str, path: str, payload: dict) -> dict:
    request = Request(
        f"{base_url.rstrip('/')}{path}",
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    with urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode('utf-8'))


def _stream_sse(base_url: str, path: str, payload: dict):
    request = Request(
        f"{base_url.rstrip('/')}{path}",
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    with urlopen(request, timeout=120) as response:
        for raw_line in response:
            line = raw_line.decode('utf-8').strip()
            if not line.startswith('data:'):
                continue
            data = line.removeprefix('data:').strip()
            if not data:
                continue
            yield json.loads(data)


st.set_page_config(page_title='Tragword Research Assistant', layout='wide')
st.title('Tragword Multi-Agent Research Assistant')

base_url = st.text_input('API Base URL', value='http://127.0.0.1:8000')
topic = st.text_area('研究题目', value='构建一个面向企业知识库的 Multi-Agent 研究助手')
mode = st.radio('运行模式', ['同步', '实时流式输出', '提纲审批模式'], horizontal=True)

if mode == '同步':
    if st.button('开始研究'):
        try:
            with st.spinner('研究中...'):
                result = _call_json_api(base_url, '/research', {'topic': topic})
            st.success('完成')
            st.markdown(result['final_report'])
        except URLError as exc:
            st.error(f'无法连接 API：{exc}')
        except Exception as exc:
            st.error(f'执行失败：{exc}')
elif mode == '实时流式输出':
    if st.button('开始研究'):
        try:
            final_report = ''
            with st.status('研究中...', expanded=True):
                for event in _stream_sse(base_url, '/research/stream', {'topic': topic}):
                    node = event.get('node', 'unknown')
                    detail = event.get('detail', {})
                    st.write(f'**{node}**: {detail}')
                    if node == 'final':
                        final_report = str(detail.get('final_report', ''))
            st.success('完成')
            st.markdown(final_report)
        except URLError as exc:
            st.error(f'无法连接 API：{exc}')
        except Exception as exc:
            st.error(f'执行失败：{exc}')
else:
    if st.button('开始提纲审批'):
        try:
            start = _call_json_api(
                base_url,
                '/research/interactive/start',
                {'topic': topic},
            )
            st.session_state['hitl_thread_id'] = start['thread_id']
            st.session_state['hitl_topic'] = start['topic']
            st.session_state['hitl_outline'] = list(start.get('sub_questions', []))
            st.session_state['hitl_final_report'] = ''
        except URLError as exc:
            st.error(f'无法连接 API：{exc}')
        except Exception as exc:
            st.error(f'开始失败：{exc}')

    thread_id = st.session_state.get('hitl_thread_id')
    outline = st.session_state.get('hitl_outline', [])
    if thread_id and outline:
        st.caption(f'Thread ID: {thread_id}')
        edited_questions: list[str] = []
        for index, question in enumerate(outline):
            edited_questions.append(
                st.text_area(
                    f'子问题 {index + 1}',
                    value=question,
                    key=f'hitl_outline_{thread_id}_{index}',
                )
            )
        if st.button('批准并继续'):
            try:
                final_report = ''
                with st.status('继续执行...', expanded=True):
                    for event in _stream_sse(
                        base_url,
                        '/research/interactive/resume',
                        {
                            'thread_id': thread_id,
                            'approved': True,
                            'sub_questions': edited_questions,
                        },
                    ):
                        node = event.get('node', 'unknown')
                        detail = event.get('detail', {})
                        st.write(f'**{node}**: {detail}')
                        if node == 'final':
                            final_report = str(detail.get('final_report', ''))
                st.session_state['hitl_final_report'] = final_report
                st.success('完成')
                st.markdown(final_report)
            except URLError as exc:
                st.error(f'无法连接 API：{exc}')
            except Exception as exc:
                st.error(f'执行失败：{exc}')
    final_report = st.session_state.get('hitl_final_report', '')
    if final_report:
        st.markdown(final_report)
