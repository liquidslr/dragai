from typing import Optional
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama


from dragai.agent.chat.llm.base import LanguageModel
from dragai.agent.chat.llm.manager import LanguageModelFactory
from dragai.agent.chat.constants import LLM_CACHE_FOLDER, LLM_CHUNK_SIZE, LLM_TEMP


@LanguageModelFactory.register("open_source")
class OpenSourceModel(LanguageModel):
    def __init__(
        self,
        model: str,
        embedding_model: str,
        api_key: Optional[str] = None,
        temperature: Optional[float] = LLM_TEMP,
        chunk_size: Optional[int] = LLM_CHUNK_SIZE,
        cache_folder: Optional[str] = LLM_CACHE_FOLDER,
    ):
        self.model = Ollama(model=model, request_timeout=3000)
        self.embeddings_model = HuggingFaceEmbedding(
            model_name=embedding_model,
            cache_folder=cache_folder,
            trust_remote_code=True,
        )

        Settings.llm = self.model
        Settings.chunk_size = chunk_size
        Settings.embed_model = self.embeddings_model
        self.temperature = temperature
        self.model = None
