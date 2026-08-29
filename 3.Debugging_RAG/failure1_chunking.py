from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from typing import List
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

def recursive_splitter() :

    text_loader = TextLoader("/Users/debjitghorai/prod_rag/sample2.txt", "utf-8")
    docs = text_loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=40
    )

    chunks = splitter.split_documents(docs)

    print(f"original length : {len(docs)}")
    print(f"number of chunks : {len(chunks)}")
    print(f"chunk sizes : {[len(c.page_content) for c in chunks]}")
    print(f"\n First chunk preview : \n{chunks[0].page_content[:200]}")

def overlap_imporatance() :
    text = "The quick broen fox jumps over the lazy dog" * 10

    #without overlap
    splitter_no_overlap = RecursiveCharacterTextSplitter(
        chunk_size = 50,
        chunk_overlap=0
    )

    splitter_normal = RecursiveCharacterTextSplitter(
        chunk_size=50,
        chunk_overlap=20
    )

    no_chunks = splitter_no_overlap.split_text(text)
    o_chunks = splitter_normal.split_text(text)

    print(f"the origoinal size : {len(text)}\n")
    print(f"number of non overlap chunks : {len(no_chunks)} \n")
    print(f"number of overlap chunks : {len(o_chunks)} \n")
    print(f"preview of no_chunks : {no_chunks[0][:200]}\n")
    print(f"preview of o_chunks : {o_chunks[0][:200]}\n")

    print("without overlap : \n")
    print(f" chunk1 end : ....{no_chunks[0][-20:]}")
    print(f" chunk2 starts : {no_chunks[1][:20]}.....\n")

    print("with overlap : \n")
    print(f" chunk1 end : ....{o_chunks[0][-20:]}")
    print(f" chunk2 starts : {o_chunks[1][:20]}.....")

if __name__ == "__main__" :
    #recursive_splitter()
    overlap_imporatance()