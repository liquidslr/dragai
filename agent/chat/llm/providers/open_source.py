from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama


from dragai.agent.chat.llm.base import LanguageModel
from dragai.agent.chat.llm.manager import LanguageModelFactory


@LanguageModelFactory.register("open_source")
class OpenSourceModel(LanguageModel):
    def __init__(
        self,
        model: str,
        embedding_model: str,
        temperature: float,
        chunk_size: int,
        cache_folder: str,
    ):
        Settings.llm = Ollama(model=model, request_timeout=3000)
        Settings.chunk_size = chunk_size
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=embedding_model, cache_folder=cache_folder
        )

        self.temperature = temperature
        self.model = None
