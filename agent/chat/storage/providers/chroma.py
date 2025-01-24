import chromadb

from llama_index.core import (
    StorageContext,
)
from llama_index.vector_stores.chroma import ChromaVectorStore


from dragai.agent.chat.storage.base import Store
from dragai.agent.chat.storage.manager import StorageFactory


@StorageFactory.register("chromadb")
class ChromaStorage(Store):
    def __init__(
        self, namespace: str, host: str = None, port: int = None, uri: str = None
    ):
        chroma_client = chromadb.EphemeralClient()
        self.chroma_collection = chroma_client.create_collection(namespace)
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )

    def get_storage_context(self) -> StorageContext:
        print("chroma")
        return self.storage_context

    def get_vector_store(self):
        return self.vector_store
