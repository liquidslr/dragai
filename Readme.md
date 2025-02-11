### DragAI: Open-source RAG agent for websites data


DragAI helps to build a Retrieval-Augmented Generation (RAG) chat agent. The agent integrates web crawling, vector storage, and language modeling to ingest, index, and interact with data collected from the web.

---

## Features

- **Web Crawling:** Fetch data from a target URL and store it locally.
- **Vector Store:** Create and manage a vector database for document storage and retrieval.
- **Language Model Integration:** Use an OpenAI-powered model (e.g., GPT-4) or any open source model using Ollama for generating responses.
- **Document Parsing & Indexing:** Convert input files into documents and index them for efficient search and retrieval.
- **RAG Agent:** A retrieval-augmented generation agent that integrates the vector store and language model to facilitate chat interactions with indexed documents.
- **Structered Data Generation:** Turn your documents into strcutred data using pydantic data models.

---

## Example Usage
```python
import os
import asyncio
import nest_asyncio
from dragai.agent.chat.llm import LLMManager
from dragai.agent.chat.storage import StoreManager
from dragai.agent.chat.index import IndexManager
from dragai.agent.chat.parsing import Document
from dragai.agent.chat import RAGAgent
from dragai.agent.crawler import WebCrawler

url = "https://example.com"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Crawl the web for data
crawler = WebCrawler()
await asyncio.run(
    crawler.get_page_data(
        url=url,
        folder="data",
    )
)

# Set OpenAI API key in the environment and apply nest_asyncio
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
nest_asyncio.apply()

# Create a vector store using ChromaDB
storage = StoreManager.get_or_create(
    store_type="chromadb",
    namespace=namespace,
    uri=uri,
)

# Create a language model instance
llm = LLMManager.create(
    model_type="openai",
    model="gpt-4o-mini",
    api_key=OPENAI_API_KEY,
    embedding_model="thenlper/gte-base",  # optional embedding model
)

# Read the documents from the specified directory 
input_files_dir = "data"  
document = Document(input_files=input_files_dir, llm=llm)

# Create a vector index for the documents
index = IndexManager.create(
    store_type="chromadb",
    storage=storage,
    namespace=namespace
)

# Create a RAG agent integrating LLM, storage, and index
agent = RAGAgent(llm=llm, storage=storage, index=index)

# (Optional) Index the documents if not already indexed
# agent.create_index()

# Interact with the agent via a chat prompt
prompt = "Your query or prompt here"
res = asyncio.run(agent.run(prompt))
print(res)

```
