import os
from dotenv import load_dotenv
import llama_index


load_dotenv()

os.environ = os.getenv("OPENAI_API_KEY")

from llama_index import VectorStoreIndex, SimpleDirectoryReader
documents = SimpleDirectoryReader("data").load_data()