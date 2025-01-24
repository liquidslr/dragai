from abc import ABC, abstractmethod
from typing import Type, Dict

import chromadb
from pymongo import MongoClient

from llama_index.core import (
    StorageContext,
)
from llama_index.storage.docstore.redis import RedisDocumentStore
from llama_index.storage.index_store.redis import RedisIndexStore
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.storage.docstore.mongodb import MongoDocumentStore
from llama_index.storage.index_store.mongodb import MongoIndexStore
from llama_index.storage.kvstore.redis import RedisKVStore as RedisCache

from dragai.agent.chat.config import Config


class Store(ABC):
    @abstractmethod
    def get_storage_context(self):
        pass


class StorageFactory:
    _registry: Dict[str, Type[Store]] = {}

    @classmethod
    def register(cls, store_type: str):
        def inner_wrapper(wrapped_class: Type[Store]):
            cls._registry[store_type.lower()] = wrapped_class
            return wrapped_class

        return inner_wrapper


class StoreManager:
    @staticmethod
    def create(store_type: str, **kwargs) -> Store:
        model_cls = StorageFactory._registry.get(store_type.lower())
        if not model_cls:
            raise ValueError(f"Model type '{store_type}' not registered")
        return model_cls(**kwargs)


@StorageFactory.register("chromadb")
class ChromaStorage(Store):
    def __init__(self, collection_name: str, config: Config):
        chroma_client = chromadb.EphemeralClient()
        self.chroma_collection = chroma_client.create_collection(collection_name)
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )

    def get_storage_context(self) -> StorageContext:
        return self.storage_context

    def get_vector_store(self):
        return self.vector_store


@StorageFactory.register("redis")
class RedisStore(Store):
    def __init__(self, host: str, port: int, namespace: str, uri: str = None):
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


@StorageFactory.register("mongodb")
class MongoDBStore(Store):
    def __init__(self, host: str, port: int, namespace: str, uri: str = None):

        if uri:
            self.docstore = MongoDocumentStore.from_uri(
                db_name="dragai", uri=uri, namespace=namespace
            )
            self.index_store = MongoIndexStore.from_uri(
                db_name="dragai", uri=uri, namespace=namespace
            )
        else:
            self.docstore = MongoDocumentStore.from_host_and_port(
                db_name="dragai", host=host, port=port, namespace=namespace
            )
            self.index_store = MongoIndexStore.from_host_and_port(
                db_name="dragai", host=host, port=port, namespace=namespace
            )

        self.storage_context = StorageContext.from_defaults(
            docstore=self.docstore, index_store=self.index_store
        )

        if uri:
            self.cache = MongoClient(uri)
        else:
            self.cache = MongoClient(host=host, port=port)

        db_name = "dragai"
        collection = "dragai"
        self.db = self.cache[db_name]
        self.collection = self.db[collection]

    def get_storage_context(self) -> StorageContext:
        return self.storage_context
