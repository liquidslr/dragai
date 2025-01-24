from pymongo import MongoClient

from llama_index.core import (
    StorageContext,
)
from llama_index.storage.docstore.mongodb import MongoDocumentStore
from llama_index.storage.index_store.mongodb import MongoIndexStore


from dragai.agent.chat.storage.base import Store
from dragai.agent.chat.storage.manager import StorageFactory


@StorageFactory.register("mongodb")
class MongoDBStore(Store):
    def __init__(self, namespace: str, host: str, port: int, uri: str = None):

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
