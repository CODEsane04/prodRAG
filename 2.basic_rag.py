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

load_dotenv()
llm = ChatGoogleGenerativeAI(
    model="gemma-4-31b-it",
    temperature=0.1
)
embedding_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

def load_docs() -> list :
    gpt_text_loader = TextLoader("sample_doc.txt", encoding="utf=8")
    claude_text_loader = TextLoader("sample2.txt", encoding="utf-8")

    gpt_text = gpt_text_loader.load()
    claude_text = claude_text_loader.load()

    return [gpt_text, claude_text]

def chunk_docs(doc_list) -> list :

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=160,
        chunk_overlap=32
    )

    all_chunks = []

    for docs in doc_list :
        chunks = splitter.split_documents(docs)
        all_chunks = all_chunks + chunks

    return all_chunks


if __name__ == "__main__" :

    doc_list = load_docs()
    chunks = chunk_docs(doc_list)

    vector_store = Chroma.from_documents(
        documents=chunks, embedding=embedding_model
    )

    user_query = ["why are claude models slow? if there a fix?", "what is chatGPT? how many users does it have?", "summarize everythng that you know about claude AI"]

    rag_context = vector_store.similarity_search(
        query=user_query, k=3
    )

    prompt = PromptTemplate(
        template="here is the user query : {query} and this is the available context : {context}. Your job is to answer the user query in a polite and respective manner & only and only answer if the appropriate context is availabel in the context provided, if the context does not have enough info to answer the query, simply say 'I cannot answer this question at this moment...' ",
        input_variables=["context", "query"]
    )

    parser = StrOutputParser()

    chain = prompt | llm | parser

    for queries in user_query :
        response = chain.invoke({
            "context" : rag_context,
            "query" : queries
        })

        print(response)
