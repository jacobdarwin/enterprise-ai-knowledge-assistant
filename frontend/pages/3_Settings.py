import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # noqa: E402
from frontend.api_client import BACKEND_URL, APIError, health_check  # noqa: E402

st.set_page_config(page_title="Settings", page_icon="⚙️", layout="wide")
st.title("⚙️ Settings")

st.subheader("Backend connection")
col1, col2 = st.columns(2)
col1.text_input("Backend URL", value=BACKEND_URL, disabled=True)
try:
    health_check()
    col2.success("Backend reachable ✅")
except APIError as exc:
    col2.error(f"Backend unreachable: {exc.detail}")
except Exception:
    col2.error("Backend unreachable — is the FastAPI server running?")

st.divider()
st.subheader("Retrieval preferences (this session)")
st.caption(
    "These apply to your next chat message only — server-side defaults "
    "(chunk size, hybrid search weights, similarity threshold) live in the "
    "backend's .env file and require a server restart to change."
)
st.session_state.setdefault("top_k_override", 10)
st.session_state.top_k_override = st.slider(
    "Number of chunks to retrieve (top_k)", min_value=1, max_value=20, value=st.session_state.top_k_override
)

st.divider()
st.subheader("Observability")
langsmith_project = os.environ.get("LANGCHAIN_PROJECT", "enterprise-rag")
st.markdown(
    f"Traces for every chat request (token usage, latency, per-node execution, retrieval "
    f"performance) are logged to LangSmith under project **`{langsmith_project}`**."
)
st.link_button("Open LangSmith traces ↗", "https://smith.langchain.com")

st.divider()
st.subheader("MCP Servers")
try:
    from frontend.api_client import get_metrics

    mcp_servers = get_metrics().get("mcp_servers", {})
    for name, info in mcp_servers.items():
        badge = "🟢 enabled" if info["enabled"] else "⚪ disabled (no token configured)"
        st.markdown(f"**{name}** — {badge}")
        st.caption(info["description"])
except APIError as exc:
    st.warning(f"Could not fetch MCP server status: {exc.detail}")
