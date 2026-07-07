import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # noqa: E402
from frontend.api_client import APIError, get_history  # noqa: E402

st.set_page_config(page_title="Conversation History", page_icon="🕑", layout="wide")
st.title("🕑 Conversation History")

try:
    result = get_history()
    conversations = result.get("conversations", [])
except APIError as exc:
    conversations = []
    st.warning(f"Could not reach backend: {exc.detail}")

if not conversations:
    st.info("No conversations yet — start chatting on the main page.")
else:
    selected = st.selectbox("Select a conversation", options=conversations)
    if selected:
        detail = get_history(conversation_id=selected)
        for msg in detail.get("messages", []):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("citations"):
                    with st.expander(f"📎 {len(msg['citations'])} source(s)"):
                        for c in msg["citations"]:
                            page_str = f"page {c['page']}" if c.get("page") is not None else "no page info"
                            st.markdown(f"**{c['filename']}** ({page_str})")
                            st.caption(c["snippet"])
