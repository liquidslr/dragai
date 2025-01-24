from abc import ABC, abstractmethod


class Store(ABC):
    @abstractmethod
    def get_storage_context(self):
        pass
