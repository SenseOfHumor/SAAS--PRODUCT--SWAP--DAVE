import os
from dotenv import load_dotenv
from datasets import load_dataset
import pandas as pd
import numpy as np
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
import argparse
#from langchain import OpenAIEmbeddings

from langchain_community.vectorstores.chroma import Chroma
from langchain.prompts.chat import ChatPromptTemplate
import os
import shutil

from langchain_community.document_loaders import DirectoryLoader

load_dotenv()
DATA_PATH = "data/pdfs"
CHROMA_PATH = "chroma"


def main():
    generate_data_store()


def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)

def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob = "*.md")
    documents = loader.load()
    return documents

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 256, 
                                                   chunk_overlap = 20,
                                                   length_function = len,
                                                   add_start_index = True,
                                                   )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks")

    document = chunks[10]
    print(document.page_content)
    print(document.metadata)

def save_to_chroma(chunks: list[Document]):
    ## clear the db first
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    
    ## new DB from the docs
    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    )

    #db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}")

    return chunks

# split_text(load_documents())