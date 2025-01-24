from typing import Type, Dict

from .base import LanguageModel


class LanguageModelFactory:
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
        model_cls = LanguageModelFactory._registry.get(model_type.lower())
        if not model_cls:
            raise ValueError(f"Model type '{model_type}' not registered")
        return model_cls(**kwargs)
