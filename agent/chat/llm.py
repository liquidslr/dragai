from typing import Type, Dict


from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI


class LanguageModel:
    pass


class ModelFactory:
    _registry: Dict[str, Type[LanguageModel]] = {}

    @classmethod
    def register(cls, model_type: str):
        def inner_wrapper(wrapped_class: Type[LanguageModel]):
            cls._registry[model_type.lower()] = wrapped_class
            return wrapped_class

        return inner_wrapper


class LLMManager:
    @staticmethod
    def create(model_type: str, **kwargs) -> LanguageModel:
        model_cls = ModelFactory._registry.get(model_type.lower())
        if not model_cls:
            raise ValueError(f"Model type '{model_type}' not registered")
        return model_cls(**kwargs)


@ModelFactory.register("openai")
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
        Settings.chunk_size = chunk_size
        if embedding_model:
            Settings.embed_model = HuggingFaceEmbedding(
                model_name=embedding_model, cache_folder=cache_folder
            )


@ModelFactory.register("open_source")
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
