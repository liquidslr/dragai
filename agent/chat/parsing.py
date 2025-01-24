from typing import List

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import SimpleDirectoryReader


class DocumentHandler:
    def __init__(self, input_files: List[str]):
        self.reader = SimpleDirectoryReader(input_files=input_files)
        self.parser = SentenceSplitter()

    def load_nodes(self):
        documents = self.reader.load_data()
        return self.parser.get_nodes_from_documents(documents)
