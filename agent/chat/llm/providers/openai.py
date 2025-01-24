from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI

from dragai.agent.chat.llm.base import LanguageModel
from dragai.agent.chat.llm.manager import LanguageModelFactory


@LanguageModelFactory.register("openai")
class OpenAIModel(LanguageModel):
    def __init__(
        self,
        api_key: str,
        model: str,
        cache_folder: str,
        temperature: float,
        chunk_size: int = 1024,
        embedding_model: str = None,
    ):
        Settings.llm = OpenAI(api_key=api_key, temperature=temperature, model=model)
        # Settings.embed_mode = OpenAIEmbedding()
        Settings.chunk_size = chunk_size
        if embedding_model:
            Settings.embed_model = HuggingFaceEmbedding(
                model_name=embedding_model, cache_folder=cache_folder
            )
