from dragai.agent.chat import Config, Logger
from dragai.agent.chat.index import IndexManager

from dragai.agent.chat.storage import StoreManager
from dragai.agent.chat.llm import LLMManager
from dragai.agent.chat.parsing import DocumentHandler


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

    def create_index(self, key_name: str):
        self.index_manager.create_index(key_name)

    async def run(self, prompt: str, key_name: str) -> str:
        if self.index_manager.index:
            _ = self.index_manager.index
        else:
            self.index_manager.load_index(key_name)

        return await self.index_manager.chat(prompt)
