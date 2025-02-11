from typing import Optional

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.program.openai import OpenAIPydanticProgram
from llama_index.core import ChatPromptTemplate

from dragai.agent.chat.llm.base import LanguageModel
from dragai.agent.chat.llm.manager import LanguageModelFactory
from dragai.agent.chat.constants import LLM_CACHE_FOLDER, LLM_CHUNK_SIZE, LLM_TEMP


@LanguageModelFactory.register("openai")
class OpenAIModel(LanguageModel):
    def __init__(
        self,
        api_key: str,
        model: str,
        embedding_model: Optional[str] = None,
        temperature: Optional[float] = LLM_TEMP,
        chunk_size: Optional[int] = LLM_CHUNK_SIZE,
        cache_folder: Optional[str] = LLM_CACHE_FOLDER,
    ):
        self.model = OpenAI(api_key=api_key, temperature=temperature, model=model)

        if embedding_model:
            self.embeddings_model = HuggingFaceEmbedding(
                model_name=embedding_model,
                cache_folder=cache_folder,
                trust_remote_code=True,
            )

        Settings.llm = self.model
        Settings.chunk_size = chunk_size
        if embedding_model:
            Settings.embed_model = self.embeddings_model

    def complete(self, prompt: str) -> str:
        return self.model.complete(prompt)

    def extract_data(
        self, prompt: ChatPromptTemplate, response_format: any, data: str
    ) -> str:
        program = OpenAIPydanticProgram.from_defaults(
            output_cls=response_format,
            llm=self.model,
            prompt=prompt,
            verbose=True,
        )
        output = program()
        return output.dict()
