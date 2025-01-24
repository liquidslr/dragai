from typing import List

from llama_index.core import (
    SimpleDirectoryReader,
)
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.tools import QueryEngineTool
from llama_index.core.tools.types import ToolMetadata

from dragai.agent.chat import Config, StoreManager, Logger
from dragai.agent.chat.index import IndexManager
from dragai.agent.chat.llm import LLMManager


class DocumentHandler:
    def __init__(self, input_files: List[str]):
        self.reader = SimpleDirectoryReader(input_files=input_files)
        self.parser = SentenceSplitter()

    def get_nodes(self):
        documents = self.reader.load_data()
        nodes = self.parser.get_nodes_from_documents(documents)
        return nodes

    def get_documents(self):
        documents = self.reader.load_data()
        return documents


class AgentManager:
    def __init__(self, index, api_key: str):
        query_tool = QueryEngineTool(
            query_engine=index.as_query_engine(),
            metadata=ToolMetadata(
                name="agent",
                description=(
                    "Answers questions related to the data."
                    "Use a detailed plain text question as input to the tool."
                ),
            ),
        )
        self.query_engine = index.as_query_engine()
        self.agent = OpenAIAgent.from_tools(
            [query_tool], api_key=api_key, verbose=False
        )

    async def chat(self, prompt: str) -> str:
        return self.agent.chat(prompt)


class RAGAgent:
    def __init__(self, config: Config):
        Logger.setup()
        self.config = config

        self.storage_manager = None
        self.storage_manager = StoreManager.create(
            store_type=self.config.store_type,
            host=config.STORE_HOST,
            port=config.STORE_PORT,
            namespace=config.NAMESPACE,
            uri=config.URI,
        )

        self.document_handler = DocumentHandler(self.config.INPUT_FILES)
        self.index_manager = IndexManager.create(
            store_type=self.config.store_type,
            storage_manager=self.storage_manager,
            document_handler=self.document_handler,
            config=self.config,
        )

        self.llm_manager = LLMManager.create(
            model_type=config.MODEL_TYPE,
            api_key=config.OPENAI_API_KEY,
            model=config.MODEL,
            temperature=config.TEMPERATURE,
            chunk_size=config.CHUNK_SIZE,
            embedding_model=config.EMBEDDING_MODEL,
            cache_folder="./store/",
        )

        self.agent_manager = None

    def setup_agent(self, index):
        self.agent_manager = AgentManager(index, self.config.OPENAI_API_KEY)

    def create_index(self, key_name: str):
        self.index_manager.create_index(key_name)

    async def run(self, prompt: str, key_name: str) -> str:
        if self.index_manager.index:
            index = self.index_manager.index
        else:
            index = self.index_manager.load_index(key_name)

        self.setup_agent(index)
        return await self.agent_manager.chat(prompt)
