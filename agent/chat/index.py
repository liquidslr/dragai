import os
import json
from abc import ABC, abstractmethod
from typing import Type, Dict

from llama_index.core import (
    SummaryIndex,
    load_index_from_storage,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter



class IndexBase(ABC):
    @abstractmethod
    def create_index(self, key_name: str):
        pass


class IndexFactory:
    _registry: Dict[str, Type[IndexBase]] = {}

    @classmethod
    def register(cls, store_type: str):
        def inner_wrapper(wrapped_class: Type[Index]):
            cls._registry[store_type.lower()] = wrapped_class
            return wrapped_class

        return inner_wrapper


class IndexManager:
    @staticmethod
    def create(store_type: str, **kwargs) -> IndexBase:
        model_cls = IndexFactory._registry.get(store_type.lower())
        if not model_cls:
            raise ValueError(f"Model type '{store_type}' not registered")
        return model_cls(**kwargs)


class Index(IndexBase):
    @classmethod
    def store_index_id(self, file_path, index):
        if os.path.exists(file_path):
            try:
                with open(file_path, "r") as file:
                    data = json.load(file)
            except json.JSONDecodeError:
                data = {}
        else:
            data = {}

        data.update(index)

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            json.dump(data, file, indent=4)

    @classmethod
    def load_index_id(self, file_path, key_name: str):
        try:
            with open(file_path, "r") as file:
                data = json.load(file)
            index_id = data.get(key_name)
            if index_id is None:
                print(f"Error: The key '{key_name}' does not exist in {file_path}.")
            return index_id
        except FileNotFoundError:
            print(f"Error: The file at {file_path} does not exist.")
            return None


@IndexFactory.register("chroma")
class ChromaIndex(Index):
    def __init__(self, storage_manager, document_handler, config):
        self.storage_manager = storage_manager
        self.document_handler = document_handler
        self.index = None
        self.config = config
        self.parser = SentenceSplitter()
        self.storage_context = storage_manager.get_storage_context()

    def create_index(self, key_name: str):
        documents = self.document_handler.get_documents()
        index = VectorStoreIndex.from_documents(
            documents, storage_context=self.storage_context
        )
        self.index = index

        file_path = f"./storage/{key_name}/chroma_index_id.json"
        data = {key_name: self.index.index_id}
        self.store_index_id(file_path, data)

    def load_index(self, key_name):
        vector_store = self.storage_manager.get_vector_store()
        print(vector_store, "vector_store")
        index = VectorStoreIndex.from_vector_store(vector_store)
        print(index, "index")

        self.index = index
        return self.index


@IndexFactory.register("disk")
class DiskIndex(Index):
    def __init__(self, storage_manager, document_handler, config):
        self.storage_manager = storage_manager
        self.document_handler = document_handler
        self.index = None
        self.config = config
        self.parser = SentenceSplitter()
        self.storage_context = StorageContext.from_defaults()

    def create_index(self, key_name: str):
        documents = self.document_handler.get_documents()
        index = VectorStoreIndex.from_documents(
            documents, storage_context=self.storage_context
        )
        self.storage_context.persist(persist_dir=f"./storage/{key_name}")
        self.index = index

        file_path = f"./storage/{key_name}/disk_index_id.json"
        data = {key_name: self.index.index_id}
        self.store_index_id(file_path, data)

    def load_index(self, key_name):

        self.storage_context = StorageContext.from_defaults(
            persist_dir=f"./storage/{key_name}"
        )
        self.index = load_index_from_storage(self.storage_context)
        return self.index


@IndexFactory.register("redis")
class RedisIndex(Index):
    def __init__(self, storage_manager, document_handler, config):
        self.storage_manager = storage_manager
        self.document_handler = document_handler
        self.index = None
        self.config = config
        self.parser = SentenceSplitter()
        self.storage_context = self.storage_manager.get_storage_context()

    def create_index(self, key_name: str):
        nodes = self.document_handler.get_nodes()
        summary_index = SummaryIndex(nodes, storage_context=self.storage_context)
        self.index = summary_index

        file_path = f"./storage/{key_name}/redis_index_id.json"
        data = {key_name: self.index.index_id}
        self.store_index_id(file_path, data)

    def load_index(self, key_name):
        file_path = f"./storage/{key_name}/redis_index_id.json"
        index_id = self.load_index_id(file_path, key_name)
        if index_id:
            self.index = load_index_from_storage(
                storage_context=self.storage_context, index_id=index_id
            )
        return self.index
