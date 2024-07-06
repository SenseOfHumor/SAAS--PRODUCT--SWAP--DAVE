import os
from dotenv import load_dotenv
from datasets import load_dataset
import pandas as pd
import numpy as np
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
#from langchain import OpenAIEmbeddings

def main():
    #CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The text to search for")
    args = parser.parse_args()
    query_text = args.query_text

    ## prep the db
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    ##search the db
    results = db.similarity_search_with_relevance_score(query_text, k=3) ##k is the number of results
    List[Tuple[Document, float]]