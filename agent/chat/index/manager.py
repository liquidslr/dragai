from typing import Dict, Type

from .base import IndexBase


class IndexFactory:
    _registry: Dict[str, Type[IndexBase]] = {}

    @classmethod
    def register(cls, store_type: str):
        def decorator(wrapped_class: Type[IndexBase]):
            cls._registry[store_type.lower()] = wrapped_class
            return wrapped_class

        return decorator

    @classmethod
    def create(cls, store_type: str, **kwargs) -> IndexBase:
        model_cls = cls._registry.get(store_type.lower())
        if not model_cls:
            raise ValueError(f"Model type '{store_type}' is not registered.")
        try:
            return model_cls(**kwargs)
        except TypeError as e:
            raise ValueError(f"Error initializing '{store_type}' index: {e}")


class IndexManager:
    @staticmethod
    def create(store_type: str, **kwargs) -> IndexBase:
        return IndexFactory.create(store_type, **kwargs)
