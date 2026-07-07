import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # noqa: E402
from frontend.api_client import APIError, delete_document, list_documents, upload_document  # noqa: E402

st.set_page_config(page_title="Upload Documents", page_icon="📁", layout="wide")
st.title("📁 Upload Documents")
st.caption("Supported types: PDF, DOCX, TXT, Markdown, CSV, JSON")

uploaded_file = st.file_uploader(
    "Choose a file to ingest",
    type=["pdf", "docx", "txt", "md", "csv", "json"],
)

if uploaded_file is not None:
    if st.button("Upload & Index", type="primary"):
        with st.spinner(f"Extracting, chunking, and embedding '{uploaded_file.name}'..."):
            try:
                result = upload_document(
                    filename=uploaded_file.name,
                    file_bytes=uploaded_file.getvalue(),
                    content_type=uploaded_file.type or "application/octet-stream",
                )
                if result["status"] == "indexed":
                    st.success(f"Indexed '{result['filename']}' into {result['num_chunks']} chunks.")
                else:
                    st.error(f"Ingestion failed: {result.get('error_message', 'unknown error')}")
            except APIError as exc:
                st.error(f"Upload failed: {exc.detail}")

st.divider()
st.subheader("Indexed Documents")

try:
    documents = list_documents()
except APIError as exc:
    documents = []
    st.warning(f"Could not reach backend: {exc.detail}")

if not documents:
    st.info("No documents uploaded yet.")
else:
    for doc in documents:
        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
        col1.markdown(f"**{doc['filename']}**")
        status_emoji = {"indexed": "✅", "processing": "⏳", "failed": "❌", "uploaded": "📥"}
        col2.markdown(f"{status_emoji.get(doc['status'], '')} {doc['status']}")
        col3.markdown(f"{doc['num_chunks']} chunks")
        if col4.button("Delete", key=f"del_{doc['document_id']}"):
            try:
                delete_document(doc["document_id"])
                st.success(f"Deleted '{doc['filename']}'")
                st.rerun()
            except APIError as exc:
                st.error(f"Delete failed: {exc.detail}")
