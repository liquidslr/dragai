from typing import Type, Dict
from .base import Store


class StorageFactory:
    _registry: Dict[str, Type[Store]] = {}

    @classmethod
    def register(cls, store_type: str):
        def decorator(wrapped_class: Type[Store]):
            cls._registry[store_type.lower()] = wrapped_class
            return wrapped_class

        return decorator

    @classmethod
    def get_or_create(cls, store_type: str, **kwargs) -> Store:
        model_cls = cls._registry.get(store_type.lower())
        if not model_cls:
            raise ValueError(f"Model type '{store_type}' is not registered.")
        try:
            return model_cls(**kwargs)
        except TypeError as e:
            raise ValueError(f"Error initializing '{store_type}' index: {e}")


class StoreManager:
    @staticmethod
    def get_or_create(store_type: str, **kwargs) -> Store:
        return StorageFactory.get_or_create(store_type, **kwargs)
