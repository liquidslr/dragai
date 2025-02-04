import chromadb
from typing import Optional

from llama_index.core import (
    StorageContext,
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.vector_stores.simple import SimpleVectorStore

from dragai.agent.chat.storage.base import Store
from dragai.agent.chat.storage.manager import StorageFactory
from dragai.agent.chat.constants import PERSISITENT_STORAGE


@StorageFactory.register("chromadb")
class ChromaStorage(Store):
    def __init__(
        self,
        namespace: str,
        host: str = None,
        port: int = None,
        persist_dir: str = PERSISITENT_STORAGE,
        load_persistent: Optional[bool] = False,
        uri: Optional[str] = None,
    ):
        if load_persistent:
            self.storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
            self.vector_store = (
                SimpleVectorStore.from_persist_dir(persist_dir=persist_dir),
            )
        else:
            db = chromadb.PersistentClient(uri)
            self.chroma_collection = db.get_or_create_collection(namespace)

            self.vector_store = ChromaVectorStore(
                chroma_collection=self.chroma_collection
            )
            self.storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )
            self.storage_context.persist(persist_dir=persist_dir)

    def get_storage_context(self) -> StorageContext:
        return self.storage_context

    def get_vector_store(self):
        return self.vector_store
