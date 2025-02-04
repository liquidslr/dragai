from typing import List, Optional

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import SimpleDirectoryReader
from llama_index.core.extractors import (
    SummaryExtractor,
    TitleExtractor,
    KeywordExtractor,
)
from llama_index.core.ingestion import IngestionPipeline

from dragai.agent.chat.llm.manager import LLMManager


class MetadataExtractor:
    def __init__(self, llm_manager: LLMManager):
        self.transformations = [
            SentenceSplitter(),
            TitleExtractor(nodes=5, llm=llm_manager.model),
            SummaryExtractor(summaries=["self"], llm=llm_manager.model),
            KeywordExtractor(keywords=10, llm=llm_manager.model),
        ]

        self.pipeline = IngestionPipeline(transformations=self.transformations)

    def add_metadata(self, documents):
        """Run the ingestion pipeline on the documents to add metadata."""

        return self.pipeline.run(documents=documents)


class Document:
    def __init__(
        self,
        input_files: List[str] | str,
        llm: LLMManager,
        add_metadata: Optional[bool] = False,
    ):
        if isinstance(input_files, list):
            self.reader = SimpleDirectoryReader(input_files=input_files)
        else:
            self.reader = SimpleDirectoryReader(input_dir=input_files)
        self.parser = SentenceSplitter()

        self.add_metadata = add_metadata
        if self.add_metadata:
            self.metadata_extractor = MetadataExtractor(llm)

    def create_nodes(self):
        """Create nodes from documents, optionally adding metadata."""

        documents = self.reader.load_data()

        if self.add_metadata:
            nodes = self.metadata_extractor.add_metadata(documents)
        else:
            nodes = self.parser.get_nodes_from_documents(documents, show_progress=True)
        return nodes

    def get_nodes(self):
        """Get nodes from documents without metadata extraction."""

        documents = self.reader.load_data()
        nodes = self.parser.get_nodes_from_documents(documents, show_progress=True)
        return nodes

    def get_documents(self):
        """Load and return the documents from the input files."""

        documents = self.reader.load_data()
        return documents
