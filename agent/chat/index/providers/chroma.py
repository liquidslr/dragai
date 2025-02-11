from typing import Optional
from enum import Enum
from llama_index.core import VectorStoreIndex
from dragai.agent.chat.index.base import IndexBase
from dragai.agent.chat.index.manager import IndexFactory
from dragai.agent.chat.storage.manager import StoreManager
from dragai.agent.chat.parsing import Document
from dragai.agent.chat.index.retrievers import (
    auto_query_engine,
    custom_query_engine,
    fusion_query_engine,
    rerank_query_engine,
)
from .type import IndexType, QueryEngineType


@IndexFactory.register("chromadb")
class ChromaIndex(IndexBase):
    def __init__(
        self,
        storage: StoreManager,
        index_type: IndexType = IndexType.VECTOR.value,
        namespace: Optional[str] = None,
        document: Optional[Document] = None,
        query_engine: str = QueryEngineType.AUTO.value,
    ):
        # Initialize with storage context and document handler.
        self.document_handler = document
        self.storage_manager = storage
        self.storage_context = storage.get_storage_context()
        self.index_type = index_type
        self.namespace = namespace
        self.index = None
        self.query_engine = query_engine

    def create_index(self):
        """Create an index from document nodes."""
        try:
            if self.index_type == IndexType.SUMMARY:
                # nodes = self.document_handler.get_nodes()
                # index = SummaryIndex(nodes, storage_context=self.storage_context)
                raise NotImplementedError("Not implemented")
            else:
                nodes = self.document_handler.create_nodes()
                index = VectorStoreIndex(
                    nodes,
                    storage_context=self.storage_context,
                    show_progress=True,
                )
            self.index = index
        except Exception as e:
            print(f"ChromaIndex creation failed: {e}")

    def _load_index(self, key_name: str = None):
        """Load index from the vector store."""
        try:
            if self.index_type == IndexType.SUMMARY:
                raise NotImplementedError("Not implemented")
            else:
                index = VectorStoreIndex.from_vector_store(
                    self.storage_manager.vector_store
                )
            self.index = index
        except Exception as e:
            print(f"ChromaIndex loading failed: {e}")
            return None

    def _get_query_engine(self, top_k: int):
        """Get the query engine based on index type."""
        print(self.query_engine, QueryEngineType.AUTO.value, "self query engine")
        if self.query_engine == QueryEngineType.AUTO.value:
            return auto_query_engine(
                self.storage_context, self.document_handler, self.index, top_k
            )
        elif self.query_engine == QueryEngineType.BM25.value:
            return fusion_query_engine(
                self.storage_context, self.document_handler, self.index, top_k
            )
        elif self.query_engine == QueryEngineType.RERANK.value:
            return rerank_query_engine(
                self.storage_context, self.document_handler, self.index, top_k
            )
        else:
            return custom_query_engine(
                self.storage_context, self.document_handler, self.index, top_k
            )

    def _setup_query_agent(self, top_k: int):
        """Setup the query engine based on index type."""
        try:
            if self.index_type == IndexType.SUMMARY:
                return self.index.as_query_engine()
            else:
                query_engine, _ = self._get_query_engine(top_k)

                return query_engine
        except Exception as e:
            print(f"Setup query agent failed: {e}")
            return None

    async def chat(self, prompt: str, top_k: int = 5) -> str:
        """Query the index asynchronously with the given prompt."""
        try:
            if self.index is None:
                self._load_index(self.namespace)

            query_engine = self._setup_query_agent(top_k=top_k)

            if query_engine:
                return query_engine.query(prompt)
            return "Query engine setup failed."
        except Exception as e:
            print(f"Chat failed: {e}")
            return "An error occurred during chat."

    def get_relevant_nodes(self, prompt: str, top_k: int) -> str:
        """Retrieve relevant nodes for the given prompt."""
        if self.index is None:
            self._load_index()

        _, retriever = self._get_query_engine(top_k)
        return retriever.retrieve(prompt)

    def get_documents(self):
        if self.index is None:
            self._load_index()

        return self.index.get_nodes()
