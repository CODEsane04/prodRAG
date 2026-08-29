from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_classic.retrievers import EnsembleRetriever, BM25Retriever
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

embedding_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

documents = [
    Document(
        page_content="""
        ChatGPT is a conversational AI assistant developed by OpenAI.
        It was publicly released on November 30, 2022.
        ChatGPT can answer questions, generate code, summarize text,
        and assist with many knowledge tasks.
        """,
        metadata={"topic": "AI"}
    ),

    Document(
        page_content="""
        GPT-4 is a large language model created by OpenAI.
        It provides stronger reasoning capabilities than GPT-3.5
        and supports multimodal inputs.
        """,
        metadata={"topic": "AI"}
    ),

    Document(
        page_content="""
        The employee handbook states that all full-time employees
        receive 20 days of paid time off (PTO) annually.
        PTO requests must be submitted through the HR portal.
        """,
        metadata={"topic": "HR"}
    ),

    Document(
        page_content="""
        Interns are not eligible for standard PTO benefits.
        Instead they receive five personal leave days per year.
        """,
        metadata={"topic": "HR"}
    ),

    Document(
        page_content="""
        Error code ERR_2341 occurs when the authentication token
        has expired. Users should re-authenticate to obtain
        a new access token.
        """,
        metadata={"topic": "Engineering"}
    ),

    Document(
        page_content="""
        Error code ERR_9182 indicates a database connection failure.
        Check network connectivity and database credentials.
        """,
        metadata={"topic": "Engineering"}
    ),

    Document(
        page_content="""
        Project Phoenix is the company's internal effort to
        modernize legacy infrastructure and migrate services
        to cloud-native architecture.
        """,
        metadata={"topic": "Projects"}
    ),

    Document(
        page_content="""
        The cloud migration initiative focuses on improving
        scalability, reliability, and deployment speed across
        engineering teams.
        """,
        metadata={"topic": "Projects"}
    ),

    Document(
        page_content="""
        Revenue for Q1 2025 reached $12.4 million,
        representing a 17 percent increase compared
        with the previous quarter.
        """,
        metadata={"topic": "Finance"}
    ),

    Document(
        page_content="""
        Operating expenses increased due to investments in
        infrastructure, research, and personnel expansion.
        """,
        metadata={"topic": "Finance"}
    )
]

vector_store = Chroma.from_documents(
    documents=documents, embedding=embedding_model
)

#create vector retrirver
vector_retriever = vector_store.as_retriever(
    search_kwargs={'k' : 3}
)

#create BM25 Retriever on raw text
bm25_retriever = BM25Retriever.from_documents(
    documents,
    k=3
)

#combine with ensemble
ensemble_retriver = EnsembleRetriever(
    retrievers=[bm25_retriever,vector_retriever],
    weights=[0.5,0.5] #equal weights to both
)

def test_retrievers(query, name, retriver) :
    results = retriver.invoke(query)
    print(f'\\n{name} - Query: \"{query}\"')
    for i,doc in enumerate(results[:3]) :
        preview = doc.page_content[:80] + '...'
        print(f'. {i+1}. {preview}')
    return results

test_queries = [
    "What does ERR_2341 mean?",
    "How many vacation days do full-time employees receive?",
    "Tell me about the cloud modernization project.",
    "Who gets fewer leave days, interns or full-time employees?",
    "What happened to revenue in Q1 2025?"
]

for query in test_queries :
    print('=' * 60)

    #vector only
    vector_results = test_retrievers(query, 'VECTOR', vector_retriever)

    #BM25 only
    vector_results = test_retrievers(query, 'BM25', bm25_retriever)

    #Hybrid
    vector_results = test_retrievers(query, 'HYBRID', ensemble_retriver)

