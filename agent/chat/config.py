import llama_index.core
import os


class Config:
    def __init__(
        self,
        file_path,
        llm_api_key,
        namespace,
        model,
        host=None,
        port=None,
        uri=None,
        embedding_model=None,
        model_type="openai",
        store_type="redis",
        persist_disk=False,
        phoenix_api_key=None,
    ):

        PHOENIX_API_KEY = phoenix_api_key
        os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"api_key={PHOENIX_API_KEY}"
        llama_index.core.set_global_handler(
            "arize_phoenix", endpoint="https://llamatrace.com/v1/traces"
        )

        self.URI = uri
        self.STORE_HOST = host
        self.STORE_PORT = port
        self.INPUT_FILES = file_path
        self.OPENAI_API_KEY = llm_api_key
        self.NAMESPACE = namespace
        self.MODEL = model
        self.TEMPERATURE = 0
        self.CHUNK_SIZE = 1024
        self.EMBEDDING_MODEL = embedding_model
        self.MODEL_TYPE = model_type
        self.persist_disk = persist_disk
        self.store_type = store_type
