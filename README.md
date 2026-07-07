# Enterprise Knowledge Assistant (RAG)

A production-oriented Retrieval-Augmented Generation (RAG) system designed with enterprise software engineering principles and optimized for local development on laptops with limited hardware (including systems with a 4GB GPU or CPU-only environments).

The project demonstrates modern GenAI engineering practices, including hybrid retrieval, multi-agent orchestration, observability, modular architecture, and deployment readiness. While optimized for portfolio and single-instance deployments, the architecture is designed so production components (such as PostgreSQL or Qdrant) can be integrated with minimal code changes.

## Technology Stack

* **LLM:** Groq (default) with configurable support for OpenAI, Gemini, and Anthropic
* **Vector Database:** ChromaDB (local)
* **Embeddings:** sentence-transformers (CPU-friendly)
* **Agent Orchestration:** LangGraph
* **Observability:** LangSmith
* **Backend:** FastAPI
* **Frontend:** Streamlit

---

## MCP Integration

The project includes a modular MCP (Model Context Protocol) integration layer that supports connecting external services such as:

* Local Filesystem
* SQLite
* GitHub
* Notion

The MCP layer is implemented as an extensible infrastructure component. It provides the foundation for future agent tool-calling workflows but is **not currently invoked automatically during the standard chat pipeline**. This design allows additional enterprise tools to be integrated without changing the overall application architecture.

---

## Project Scope

This project demonstrates:

* Enterprise RAG architecture
* Hybrid Retrieval (Dense + BM25 + Reciprocal Rank Fusion)
* Cross-Encoder reranking
* LangGraph workflow orchestration
* Modular multi-agent architecture
* Source-grounded answer generation
* LangSmith tracing
* FastAPI backend
* Streamlit frontend
* Docker-based deployment
* Automated testing
* CI/CD pipeline
* Clean Architecture principles

The focus of this repository is to demonstrate production-style AI application architecture and engineering best practices rather than large-scale distributed deployment.

---

## License

This project is licensed under the MIT License.

See the `LICENSE` file for details.

---

## AI-Assisted Development

This project was developed using AI-assisted software engineering tools.

The overall system architecture, feature requirements, implementation direction, integration, testing, and iterative refinement were guided by the project author. AI was used as a coding assistant to accelerate implementation and improve development productivity.

All code has been reviewed, integrated, and validated before publication.

---

## Known Limitations

Current implementation is intentionally optimized for local development and portfolio-scale deployments.

Current limitations include:

* SQLite and embedded ChromaDB are best suited for single-instance deployments.
* Multi-node deployments should replace SQLite with PostgreSQL and ChromaDB with a distributed vector database such as Qdrant or Pinecone.
* Some MCP integrations are implemented as infrastructure components and are ready for future workflow integration rather than being automatically used in every chat request.
* Embedding and reranking models are downloaded once from Hugging Face during first execution and then cached locally.


<img width="1702" height="828" alt="Screenshot 2026-07-07 152358" src="https://github.com/user-attachments/assets/9087f62b-e7ee-40e7-b043-97518417f463" />

<img width="1736" height="827" alt="Screenshot 2026-07-07 152427" src="https://github.com/user-attachments/assets/0f4a2f51-87a1-4034-a29d-ed44336dbf20" />

<img width="1658" height="812" alt="Screenshot 2026-07-07 152516" src="https://github.com/user-attachments/assets/0af358c9-7b17-4424-9033-c201595126a2" />

<img width="1638" height="801" alt="Screenshot 2026-07-07 152631" src="https://github.com/user-attachments/assets/b64fc3c5-4633-4c03-a0cf-77aab2e44b65" />




