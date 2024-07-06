import os
from dotenv import load_dotenv
from datasets import load_dataset
import pandas as pd
import numpy as np
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain.prompts.chat import ChatPromptTemplate
from langchain_openai import ChatOpenAI
#from langchain import OpenAIEmbeddings
load_dotenv()
CHROMA_PATH = "chroma"

## OpenAI response using the prompt template
PROMPT_TEMPLATE = """
Answer the question based only on teh following context:
{context}

---

Answer the question based on the above context: {query_text}
"""

def main():
    

    ## prep the db
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # ##search the db
    # results = db.similarity_search_with_relevance_score(query_text, k=3) ##k is the number of results
    # List[Tuple[Document, float]]


    query_text = input("Enter the text to search for: ")

    ## prep the db
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    ##search the db
    results = db.similarity_search_with_relevance_scores(query_text, k=3) ##k is the number of results
    if len(results) == 0 or results[0][1] < 0.7:
        print(f"No results found for '{query_text}'")


    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    #from langchain_openai import ChatPromptTemplate
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, query_text = query_text)
    print(context_text)

    model = ChatOpenAI()
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response from {response_text}\nSources: {sources}"
    print(formatted_response)

if __name__ == "__main__":
    main()






        

