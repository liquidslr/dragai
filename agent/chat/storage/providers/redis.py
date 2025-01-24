from llama_index.core import (
    StorageContext,
)

from llama_index.storage.docstore.redis import RedisDocumentStore
from llama_index.storage.index_store.redis import RedisIndexStore
from llama_index.storage.kvstore.redis import RedisKVStore as RedisCache

from dragai.agent.chat.storage.base import Store
from dragai.agent.chat.storage.manager import StorageFactory


@StorageFactory.register("redis")
class RedisStore(Store):
    def __init__(self, namespace: str, host: str, port: int, uri: str):
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
