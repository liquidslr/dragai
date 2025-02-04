from typing import Optional

from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import QueryFusionRetriever

from dragai.agent.chat.index.manager import IndexManager
from dragai.agent.chat.storage.manager import StoreManager
from dragai.agent.chat.parsing import Document


def fusion_query_engine(
    storage_context: Optional[StoreManager],
    document_handler: Optional[Document],
    index: Optional[IndexManager],
    top_k: Optional[int],
):
    # https://docs.llamaindex.ai/en/stable/understanding/querying/querying/

    retriever = QueryFusionRetriever(
        [
            index.as_retriever(similarity_top_k=top_k),
            BM25Retriever.from_defaults(
                docstore=index.docstore, similarity_top_k=top_k
            ),
        ],
        num_queries=2,
        use_async=True,
    )
    query_engine = RetrieverQueryEngine(retriever)

    return query_engine, retriever
