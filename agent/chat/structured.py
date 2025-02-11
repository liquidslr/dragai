import json
from typing import Type, Optional
from pydantic import BaseModel
from llama_index.core import Document
from llama_index.core import ChatPromptTemplate
from llama_index.core.llms import ChatMessage

from dragai.agent.chat.llm.manager import LLMManager
from dragai.agent.chat.constants import OUTPUT_PATH


class StructuredData:
    def __init__(
        self,
        documents: list[Document],
        llm: LLMManager | None = None,
        output_path: Optional[str] = OUTPUT_PATH,
        output_model: Optional[Type[BaseModel]] = None,
    ):
        """Initialize the structured response processor."""
        self.llm = llm
        self.output_path = output_path
        self.responses = []
        self.documents = documents
        self.output_model = output_model

    def process_document(self, document: Document):
        """Process a single document and return structured output."""
        prompt = self._create_prompt(document.text)

        response = self.llm.extract_data(
            prompt, response_format=self.output_model, data=document.text
        )

        try:
            # Parse response into structured format
            self.responses.append(response)
            return response
        except Exception as e:
            raise ValueError(
                f"Failed to parse LLM response into structured format: {e}"
            )

    def process_documents(self, documents: list[Document]) -> list[BaseModel]:
        """Process multiple documents and return list of structured outputs."""
        return [self.process_document(doc) for doc in documents]

    def save_responses(self) -> None:
        """Save all processed responses to JSON file if output path is specified."""
        if not self.output_path:
            raise ValueError("No output path specified")

        responses_dict = [response for response in self.responses]
        with open(self.output_path, "w") as f:
            json.dump(responses_dict, f, indent=2)

    def _create_prompt(self, text: Document) -> str:
        data = f"""Here is the data: \n" "------\n" "{text}\n" "------\n"""
        print(data, "data")
        prompt = ChatPromptTemplate(
            message_templates=[
                ChatMessage(
                    role="system",
                    content=(
                        "You are an expert assitant for summarizing and extracting data."
                    ),
                ),
                ChatMessage(
                    role="user",
                    content=(data),
                ),
            ]
        )
        return prompt
