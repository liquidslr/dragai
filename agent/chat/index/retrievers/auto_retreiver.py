from typing import Optional

from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.vector_stores import MetadataInfo, VectorStoreInfo
from llama_index.core.retrievers import VectorIndexAutoRetriever

from dragai.agent.chat.index.manager import IndexManager
from dragai.agent.chat.storage.manager import StoreManager
from dragai.agent.chat.parsing import Document


def auto_query_engine(
    storage_context: Optional[StoreManager],
    document_handler: Optional[Document],
    index: Optional[IndexManager],
    top_k: Optional[int],
):
    # https://docs.llamaindex.ai/en/stable/understanding/querying/querying/

    vector_store_info = VectorStoreInfo(
        content_info="Interview Questions",
        metadata_info=[
            MetadataInfo(
                name="excerpt_keywords",
                description="important keywords present in the document",
                type="string",
            ),
            # MetadataInfo(
            #     name="document_title",
            #     description="title of  the document",
            #     type="string",
            # ),
            MetadataInfo(
                name="section_summary",
                description="summary of the document",
                type="string",
            ),
        ],
    )

    retriever = VectorIndexAutoRetriever(
        index,
        vector_store_info=vector_store_info,
        similarity_top_k=top_k,
        empty_query_top_k=top_k,
        verbose=True,
    )

    query_engine = RetrieverQueryEngine.from_args(retriever)

    return query_engine, retriever
