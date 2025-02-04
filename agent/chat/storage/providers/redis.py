from typing import Optional
from llama_index.core import StorageContext

from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.vector_stores.simple import SimpleVectorStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.storage.docstore.redis import RedisDocumentStore
from llama_index.storage.index_store.redis import RedisIndexStore
from llama_index.storage.kvstore.redis import RedisKVStore as RedisCache

from dragai.agent.chat.storage.base import Store
from dragai.agent.chat.storage.manager import StorageFactory
from dragai.agent.chat.constants import PERSISITENT_STORAGE


@StorageFactory.register("redis")
class RedisStore(Store):
    def __init__(
        self,
        namespace: str,
        host: str,
        port: int,
        persist_dir: str = PERSISITENT_STORAGE,
        load_persistent: Optional[bool] = False,
        uri: Optional[str] = None,
    ):

        if host is None:
            raise TypeError("Missing required argument 'host'")

        if port is None:
            raise TypeError("Missing required argument 'port'")

        if load_persistent:
            self.docstore = SimpleDocumentStore.from_persist_dir(
                persist_dir=persist_dir
            )
            self.vector_store = SimpleVectorStore.from_persist_dir(
                persist_dir=persist_dir
            )
            self.index_store = SimpleIndexStore.from_persist_dir(
                persist_dir=persist_dir
            )
            self.storage_context = StorageContext.from_defaults(
                docstore=self.docstore, index_store=self.index_store
            )
        else:
            self.docstore = RedisDocumentStore.from_host_and_port(
                host=host, port=port, namespace=namespace
            )
            self.index_store = RedisIndexStore.from_host_and_port(
                host=host, port=port, namespace=namespace
            )
            self.storage_context = StorageContext.from_defaults(
                docstore=self.docstore, index_store=self.index_store
            )
            self.cache = RedisCache.from_host_and_port(host, port)

    def get_storage_context(self) -> StorageContext:
        return self.storage_context

    def add_key(self, key, value):
        self.cache.put(key, value)

    def get_val(self, key):
        return self.cache.get(key)
