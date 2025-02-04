from typing import Optional

from llama_index.core import load_index_from_storage
from llama_index.core import (
    SummaryIndex,
    load_index_from_storage,
)


from dragai.agent.chat.index.base import IndexBase
from dragai.agent.chat.index.manager import IndexFactory
from dragai.agent.chat.index.retrievers.auto_retreiver import auto_query_engine
from dragai.agent.chat.index.retrievers.bm25 import fusion_query_engine
from dragai.agent.chat.storage.manager import StoreManager
from dragai.agent.chat.parsing import Document

from .type import IndexType


@IndexFactory.register("redis")
class RedisIndex(IndexBase):
    def __init__(
        self,
        storage: StoreManager,
        index_type: IndexType = IndexType.SUMMARY,
        namespace: Optional[str] = None,
        document: Optional[Document] = None,
    ):
        self.storage_manager = storage
        self.storage_context = self.storage_manager.get_storage_context()
        self.document_handler = document
        self.namespace = namespace
        self.index_type = index_type
        self.index = None

    def create_index(self):
        """Create an index from document nodes."""
        try:
            key_name = self.namespace
            nodes = self.document_handler.get_nodes()

            if self.index_type == IndexType.SUMMARY:
                index = SummaryIndex(nodes, storage_context=self.storage_context)
            else:
                raise NotImplementedError("Not Implemented")

            self.index = index
            file_path = f"./storage/{key_name}/redis_index_id.json"
            self.store_index_id(file_path, {key_name: self.index.index_id})
        except Exception as e:
            print(f"RedisIndex creation failed: {e}")

    def _load_index(self, key_name: str):
        """Load index from the redis store."""
        try:
            file_path = f"./storage/{key_name}/redis_index_id.json"
            index_id = self.load_index_id(file_path, key_name)

            if index_id:
                if self.index_type == IndexType.SUMMARY:
                    self.index = load_index_from_storage(
                        storage_context=self.storage_context,
                        index_id=index_id,
                    )
                else:
                    raise NotImplementedError("Not Implemented")

            return self.index
        except Exception as e:
            print(f"RedisIndex loading failed: {e}")
            return None

    def _setup_query_agent(self):
        """Setup the query engine based on index type."""
        try:
            return self.index.as_query_engine()
        except Exception as e:
            print(f"Setup query agent failed: {e}")
            return None

    async def chat(self, prompt: str) -> str:
        """Query the index asynchronously with the given prompt."""
        try:
            if self.index is None:
                self._load_index(self.namespace)

            query_engine = self._setup_query_agent()

            if query_engine:
                return query_engine.query(prompt)
            return "Query engine setup failed."
        except Exception as e:
            print(f"Chat failed: {e}")
            return "An error occurred during chat."

    def get_relevant_nodes(self, prompt: str, top_k: int) -> str:
        """Retrieve relevant nodes for the given prompt."""
        if self.index is None:
            self._load_index(self.namespace)

        retriever = self.index.as_retriever()
        return retriever.retrieve(prompt)
