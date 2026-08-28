import chromadb
chroma_client = chromadb.Client()

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import uuid
from dotenv import load_dotenv
from langchain_chroma import Chroma

load_dotenv()

def chroma_basics() :
    collection_name = "test-collection"

    collection = chroma_client.create_collection(name=collection_name)


    #define text documents
    documents = [
        {"id" : "doc1", "text" : "brazil is a nice country, they have a lot of grenary"},
        {"id" : "doc2", "text" : "paris is really famous for it's bakery and luxury items but is very expensive"},
        {"id" : "doc3", "text" : "kazaksthan is an emrging tourist spot, ans it is cheap at the same time"}
    ]


    for doc in documents : 
        collection.add(
            ids=[doc["id"]],
            documents=[doc["text"]]
        )


    #define a query text
    query = "I want to know about a cheap & emerging tourist country"

    results = collection.query(
        query_texts=[query],
        n_results=1
    )

    print(type(results))

    print(results.keys())
    print(results["documents"])

def similarity_search_with_scores() :

    #loading the document
    loader = TextLoader("sample_doc.txt", encoding="utf-8")
    documents = loader.load()

    #create splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80
    )

    #split docs
    chunks = splitter.split_documents(documents)
    print(type(chunks))
    print(type(chunks[0]))

    print(f"number of chunks : {len(chunks)}")
    #print(chunks)

    #----------------------------------------------
    embedding_model = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001"
    )

    vector_store = Chroma.from_documents(
        documents=chunks, embedding=embedding_model
    )

    query = "what is chatGPT? and when was it originally released?"

    results = vector_store.similarity_search_with_score(query, k=2)
    print(results[0][1])

def metadata_filtering() :

    #text loader
    text_loader = TextLoader("sample_doc.txt", encoding='utf-8')
    doc_gpt = text_loader.load()

    text_loaders = TextLoader("sample2.txt", encoding='utf-8')
    doc_claude = text_loaders.load()

    query="what is claude and when was it fouded"

    #splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20
    )

    gpt_chunks = splitter.split_documents(doc_gpt)

    for items in gpt_chunks :
        items.metadata["topic"] = "chatGPT"

    claude_chunks = splitter.split_documents(doc_claude)
    for items in claude_chunks :
            items.metadata["topic"] = "claude"

    #embedding model
    embedder = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001"
    )

    all_chunks = gpt_chunks + claude_chunks

    vector_store = Chroma.from_documents(
        documents=all_chunks, embedding=embedder
    )

    filter_criteria = {"topic" : "claude"}
    filtered_results = vector_store.similarity_search_with_score(
         query=query, k=10, filter=filter_criteria
    )

    for doc, score in filtered_results :
        print(doc.page_content)
        print(doc.metadata)
        print("\n")




if __name__ == "__main__" :
    #similarity_search_with_scores()
    metadata_filtering()

