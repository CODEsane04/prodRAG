from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import TextLoader
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def contextual_compression() :


    text_laoder = TextLoader("/Users/debjitghorai/prod_rag/context_comp_data.txt", encoding="utf-8")
    docs = text_laoder.load()

    embedding_model = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001"
    )

    sem_chunker = SemanticChunker(
        embedding_model,
        breakpoint_threshold_type='percentile',
        breakpoint_threshold_amount=70
    )

    chunks = sem_chunker.split_documents(docs)


    llm = ChatGoogleGenerativeAI(
        model="gemma-4-31b-it",
        temperature=0.1
    )

    vector_store = Chroma.from_documents(
        embedding=embedding_model,
        documents=chunks
    )

    compressor = LLMChainExtractor.from_llm(llm)

    compression_retriver = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=vector_store.as_retriever(search_kwargs={"k" : 4})
    )

    query = "When was Atlas version 3.2 deployed and who approved it?"

    prompt = PromptTemplate(
        template="you are a query resolvin agent, the customers are gonna ask you queries. based on the provided context : {context} answer the query : {query}. if the required ans is not in the context, strictly give th response - 'I don't have much info on this' ",
        input_variables=["context", "query"]
    )

    parser = StrOutputParser()

    chain = prompt | llm | parser

    #retruieve documents

    #without compression
    print("\n without compression \n")
    retriever = vector_store.as_retriever(
        search_kwargs={"k" : 4}
    )
    non_com_res = retriever.invoke(query)
    print(non_com_res)
    resp = chain.invoke({
        "context" : non_com_res,
        "query" : query
    })
    print("\n")
    print(f"non compression response : {resp}")
    print("\n")

    #with compression
    print("\n with compression \n")
    com_res = compression_retriver.invoke(query)
    print(com_res)
    resp = chain.invoke({
        "context" : com_res,
        "query" : query
    })
    print("\n")
    print(f"compression response : {resp}")
    print("\n")

    
if __name__ == "__main__" :
    contextual_compression()



