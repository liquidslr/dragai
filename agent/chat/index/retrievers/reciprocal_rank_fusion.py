import nest_asyncio
from typing import Optional

from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex

from dragai.agent.chat.index.manager import IndexManager
from dragai.agent.chat.storage.manager import StoreManager
from dragai.agent.chat.parsing import Document


def rerank_query_engine(
    storage_context: Optional[StoreManager],
    document_handler: Optional[Document],
    index: Optional[IndexManager],
    top_k: Optional[int],
):
    # https://docs.llamaindex.ai/en/stable/examples/retrievers/reciprocal_rerank_fusion/

    nest_asyncio.apply()

    documents = document_handler.get_documents()
    splitter = SentenceSplitter(chunk_size=256)
    index = VectorStoreIndex.from_documents(documents, transformations=[splitter])

    vector_retriever = index.as_retriever(similarity_top_k=top_k)
    bm25_retriever = BM25Retriever.from_defaults(
        docstore=index.docstore, similarity_top_k=top_k
    )

    retriever = QueryFusionRetriever(
        [vector_retriever, bm25_retriever],
        similarity_top_k=top_k,
        num_queries=1,  # set this to 1 to disable query generation
        mode="reciprocal_rerank",
        use_async=True,
        verbose=True,
    )

    query_engine = RetrieverQueryEngine.from_args(retriever)
    return query_engine, retriever
