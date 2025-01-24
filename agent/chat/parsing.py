from typing import List

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import SimpleDirectoryReader
from llama_index.core.extractors import (
    SummaryExtractor,
    TitleExtractor,
    KeywordExtractor,
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core import Settings


class MetadataExtractor:
    def __init__(self):
        self.transformations = [
            SentenceSplitter(),
            TitleExtractor(nodes=5, llm=Settings.llm),
            SummaryExtractor(summaries=["prev", "self"], llm=Settings.llm),
            KeywordExtractor(keywords=10, llm=Settings.llm),
        ]

        self.pipeline = IngestionPipeline(transformations=self.transformations)

    def add_metadata(self, documents):
        return self.pipeline.run(documents=documents)


class DocumentHandler:
    def __init__(self, input_files: List[str]):
        self.reader = SimpleDirectoryReader(input_files=input_files)
        self.parser = SentenceSplitter()
        self.add_metadata = True

        if self.add_metadata:
            self.metadata_extractor = MetadataExtractor()

    def create_nodes(self):
        documents = self.reader.load_data()

        if self.add_metadata:
            nodes = self.metadata_extractor.add_metadata(documents)
        else:
            nodes = self.parser.get_nodes_from_documents(documents, show_progress=True)

        if self.add_metadata:
            print(len(nodes))
            print(nodes[0].metadata)
        return nodes

    def get_nodes(self):
        documents = self.reader.load_data()
        nodes = self.parser.get_nodes_from_documents(documents, show_progress=True)

        return nodes

    def get_documents(self):
        documents = self.reader.load_data()
        return documents
