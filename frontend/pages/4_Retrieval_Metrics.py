import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # noqa: E402
from frontend.api_client import APIError, get_metrics  # noqa: E402

st.set_page_config(page_title="Retrieval Metrics", page_icon="📊", layout="wide")
st.title("📊 Retrieval Metrics")

try:
    metrics = get_metrics()
except APIError as exc:
    st.warning(f"Could not reach backend: {exc.detail}")
    st.stop()

col1, col2, col3 = st.columns(3)
col1.metric("Total Documents", metrics["total_documents"])
col2.metric("Total Chunks Indexed", metrics["total_chunks_indexed"])
col3.metric("Vectors in Store", metrics["vectors_in_store"])

st.divider()
st.subheader("Documents by status")
status_breakdown = metrics.get("documents_by_status", {})
if status_breakdown:
    st.bar_chart(status_breakdown)
else:
    st.info("No documents uploaded yet.")

st.divider()
st.caption(
    "For per-query latency, token usage, and retrieval quality traces, see the "
    "LangSmith project link on the Settings page."
)
