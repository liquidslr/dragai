from llama_index.core import (
    load_index_from_storage,
)

from llama_index.core import (
    SummaryIndex,
    load_index_from_storage,
)

from dragai.agent.chat.index.base import IndexBase
from dragai.agent.chat.index.manager import IndexFactory


@IndexFactory.register("redis")
class RedisIndex(IndexBase):
    def __init__(self, storage_manager, document_handler):
        self.storage_manager = storage_manager
        self.document_handler = document_handler
        self.index = None

    def create_index(self, key_name: str):
        try:
            nodes = self.document_handler.get_nodes()
            summary_index = SummaryIndex(
                nodes, storage_context=self.storage_manager.get_storage_context()
            )
            self.index = summary_index
            file_path = f"./storage/{key_name}/redis_index_id.json"
            self.store_index_id(file_path, {key_name: self.index.index_id})
        except Exception as e:
            print(f"RedisIndex creation failed: {e}")

    def load_index(self, key_name: str):
        try:
            file_path = f"./storage/{key_name}/redis_index_id.json"
            index_id = self.load_index_id(file_path, key_name)
            if index_id:
                self.index = load_index_from_storage(
                    storage_context=self.storage_manager.get_storage_context(),
                    index_id=index_id,
                )
            return self.index
        except Exception as e:
            print(f"RedisIndex loading failed: {e}")
            return None

    def setup_query_agent(self):
        try:
            return self.index.as_query_engine()
        except Exception as e:
            print(f"Setup query agent failed: {e}")
            return None

    async def chat(self, prompt: str) -> str:
        try:
            query_engine = self.setup_query_agent()
            if query_engine:
                return query_engine.query(prompt)
            return "Query engine setup failed."
        except Exception as e:
            print(f"Chat failed: {e}")
            return "An error occurred during chat."
