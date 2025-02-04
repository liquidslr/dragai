from abc import ABC


class LanguageModel(ABC):
    def __init__(self, **kwargs):
        self.params = kwargs
