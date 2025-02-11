import os
from typing import Optional
import llama_index.core

from dragai.agent.chat import Logger
from dragai.agent.chat.index import IndexManager
from dragai.agent.chat.storage import StoreManager
from dragai.agent.chat.llm import LLMManager
from dragai.agent.chat.parsing import Document


class RAGAgent:
    def __init__(
        self,
        llm: LLMManager,
        storage: StoreManager,
        index: IndexManager,
        document: Optional[Document] = None,
        phoenix_api_key: Optional[str] = None,
    ):

        if llm is None:
            raise TypeError("Missing required argument 'LLM'")

        if storage is None:
            raise TypeError("Missing required arguement 'storage")

        if index is None:
            raise TypeError("Missing required arguement 'index")

        self.llm_manager = llm
        self.storage_manager = storage
        self.index_manager = index
        self.document_handler = document

        if phoenix_api_key:
            os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"api_key={phoenix_api_key}"
            llama_index.core.set_global_handler(
                "arize_phoenix", endpoint="https://llamatrace.com/v1/traces"
            )

    def create_index(self):
        """Create nodes from documents, optionally adding metadata."""
        self.index_manager.create_index()

    def get_nodes(self, prompt: str, top_k: int = 5):
        return self.index_manager.get_relevant_nodes(prompt, top_k)

    def get_documents(self):
        return self.document_handler.get_documents()

    def run(self, prompt: str) -> str:
        """Synchronous wrapper for chat"""
        import asyncio

        return asyncio.run(self.index_manager.chat(prompt))
