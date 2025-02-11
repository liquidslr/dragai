from enum import Enum


class IndexType(Enum):
    SUMMARY = 1
    VECTOR = 2
    KEYWORDTABLE = 3


class QueryEngineType(Enum):
    CUSTOM = "custom"
    AUTO = "auto"
    BM25 = "bm25"
    RERANK = "rerank"
