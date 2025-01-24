from llama_index.core import (
    VectorStoreIndex,
)
from llama_index.core.retrievers import VectorIndexAutoRetriever
from llama_index.core.vector_stores.types import MetadataInfo, VectorStoreInfo
from llama_index.core import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor

from dragai.agent.chat.index.base import IndexBase
from dragai.agent.chat.index.manager import IndexFactory


@IndexFactory.register("chromadb")
class ChromaIndex(IndexBase):
    def __init__(self, storage_manager, document_handler):
        self.storage_manager = storage_manager
        self.document_handler = document_handler
        self.index = None

    def create_index(self, key_name: str):
        try:
            nodes = self.document_handler.create_nodes()
            index = VectorStoreIndex(
                nodes,
                storage_context=self.storage_manager.get_storage_context(),
                show_progress=True,
            )
            self.index = index
            file_path = f"./storage/{key_name}/chroma_index_id.json"
            self.store_index_id(file_path, {key_name: self.index.index_id})
        except Exception as e:
            print(f"ChromaIndex creation failed: {e}")

    def load_index(self, key_name: str = None):
        try:
            nodes = self.document_handler.get_nodes()
            index = VectorStoreIndex(
                nodes, storage_context=self.storage_manager.get_storage_context()
            )
            self.index = index
            return self.index
        except Exception as e:
            print(f"ChromaIndex loading failed: {e}")
            return None

    def setup_query_agent(self):
        try:
            self.index = self.load_index()
            vector_store_info = VectorStoreInfo(
                content_info="Interview related questions and process details",
                metadata_info=[
                    MetadataInfo(
                        name="excerpt_keywords",
                        type="str",
                        description="Important keywords in the document",
                    ),
                    MetadataInfo(
                        name="document_title",
                        type="str",
                        description="Title of the document",
                    ),
                ],
            )
            retriever = VectorIndexAutoRetriever(
                self.index, vector_store_info=vector_store_info
            )
            response_synthesizer = get_response_synthesizer()
            return RetrieverQueryEngine(
                retriever=retriever,
                response_synthesizer=response_synthesizer,
                node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)],
            )
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
