"""
Enterprise RAG Assistant — Chat page (Streamlit entry point).

Run with: streamlit run frontend/app.py
Additional pages (Upload, History, Settings, Metrics) live in frontend/pages/
and are auto-discovered by Streamlit's multipage app support.
"""

import os
import sys
from uuid import uuid4

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402
from frontend.api_client import APIError, chat_stream, list_documents  # noqa: E402

st.set_page_config(page_title="Enterprise RAG Assistant", page_icon="🧠", layout="wide")

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": ..., "content": ..., "citations": [...]}

st.title("🧠 Enterprise Knowledge Assistant")
st.caption("Ask questions about your uploaded documents. Answers are grounded only in retrieved context.")

with st.sidebar:
    st.subheader("Filter by document")
    try:
        documents = list_documents()
    except APIError as exc:
        documents = []
        st.warning(f"Could not reach backend: {exc.detail}")

    doc_options = {d["filename"]: d["document_id"] for d in documents if d["status"] == "indexed"}
    selected_filenames = st.multiselect(
        "Only search within:", options=list(doc_options.keys()), default=[], help="Leave empty to search all documents."
    )
    selected_document_ids = [doc_options[f] for f in selected_filenames] or None

    st.divider()
    if st.button("🗑️ New conversation"):
        st.session_state.conversation_id = str(uuid4())
        st.session_state.messages = []
        st.rerun()

    st.caption(f"Conversation ID: `{st.session_state.conversation_id[:8]}...`")

# --- render existing conversation ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            with st.expander(f"📎 {len(msg['citations'])} source(s)"):
                for c in msg["citations"]:
                    page_str = f"page {c['page']}" if c.get("page") is not None else "no page info"
                    st.markdown(f"**{c['filename']}** ({page_str})")
                    st.caption(c["snippet"])

# --- chat input ---
if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt, "citations": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_answer = ""
        citations = []
        try:
            for event in chat_stream(
                query=prompt,
                conversation_id=st.session_state.conversation_id,
                document_ids=selected_document_ids,
                top_k=st.session_state.get("top_k_override"),
            ):
                if event["type"] == "token":
                    full_answer += event["delta"]
                    placeholder.markdown(full_answer + "▌")
                elif event["type"] == "done":
                    citations = event.get("citations", [])
            placeholder.markdown(full_answer)
            if citations:
                with st.expander(f"📎 {len(citations)} source(s)"):
                    for c in citations:
                        page_str = f"page {c['page']}" if c.get("page") is not None else "no page info"
                        st.markdown(f"**{c['filename']}** ({page_str})")
                        st.caption(c["snippet"])
        except APIError as exc:
            full_answer = f"⚠️ Backend error: {exc.detail}"
            placeholder.markdown(full_answer)

    st.session_state.messages.append({"role": "assistant", "content": full_answer, "citations": citations})
