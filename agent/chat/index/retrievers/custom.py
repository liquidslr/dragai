from typing import List, Optional

from llama_index.core.retrievers import (
    BaseRetriever,
    VectorIndexRetriever,
    KeywordTableSimpleRetriever,
)
from llama_index.core import SimpleKeywordTableIndex
from llama_index.core import QueryBundle
from llama_index.core.schema import NodeWithScore
from llama_index.core import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine


from dragai.agent.chat.index.manager import IndexManager
from dragai.agent.chat.storage.manager import StoreManager
from dragai.agent.chat.parsing import Document


class CustomRetriever(BaseRetriever):
    """Custom retriever that performs both semantic search and hybrid search."""

    # https://docs.llamaindex.ai/en/stable/examples/query_engine/CustomRetrievers/

    def __init__(
        self,
        vector_retriever: VectorIndexRetriever,
        keyword_retriever: KeywordTableSimpleRetriever,
        mode: str = "OR",
    ) -> None:
        """Init params."""

        self._vector_retriever = vector_retriever
        self._keyword_retriever = keyword_retriever

        if mode not in ("AND", "OR"):
            raise ValueError("Invalid mode.")
        self._mode = mode
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""

        vector_nodes = self._vector_retriever.retrieve(query_bundle.query_str)
        keyword_nodes = self._keyword_retriever.retrieve(query_bundle.query_str)

        vector_ids = {n.node.node_id for n in vector_nodes}
        keyword_ids = {n.node.node_id for n in keyword_nodes}

        combined_dict = {n.node.node_id: n for n in vector_nodes}
        combined_dict.update({n.node.node_id: n for n in keyword_nodes})

        if self._mode == "AND":
            retrieve_ids = vector_ids.intersection(keyword_ids)
        else:
            retrieve_ids = vector_ids.union(keyword_ids)

        retrieve_nodes = [combined_dict[rid] for rid in retrieve_ids]
        return retrieve_nodes


def custom_query_engine(
    storage_context: Optional[StoreManager],
    document_handler: Optional[Document],
    index: Optional[IndexManager],
    top_k: Optional[int],
):
    # Hack to get all the nodes
    # https://github.com/run-llama/llama_index/issues/9893
    retriever = index.as_retriever(similarity_top_k=1000)
    source_nodes = retriever.retrieve("fake")
    nodes = [x.node for x in source_nodes]

    keyword_index = SimpleKeywordTableIndex(nodes, storage_context=storage_context)
    vector_retriever = VectorIndexRetriever(index=index)

    keyword_retriever = KeywordTableSimpleRetriever(index=keyword_index)
    custom_retriever = CustomRetriever(vector_retriever, keyword_retriever)

    response_synthesizer = get_response_synthesizer()

    query_engine = RetrieverQueryEngine(
        retriever=custom_retriever,
        response_synthesizer=response_synthesizer,
    )

    return query_engine, custom_retriever
