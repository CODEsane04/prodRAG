text = '''
Artificial Intelligence (AI) is the field of building systems that can perform tasks requiring human intelligence. Modern AI systems often use machine learning, where models learn patterns from data. Large Language Models such as GPT and Gemini are trained on massive text corpora and can generate human-like responses.

Football tactics have evolved significantly over the last decade. Teams increasingly focus on pressing, positional play, and rest defence. Rest defence refers to the positioning of players during an attack so that the team remains protected against counterattacks if possession is lost.

The Roman Empire was one of the largest civilizations in human history. It began as a republic before becoming an empire under Augustus. Roman engineering achievements included roads, aqueducts, and large-scale architectural projects.

Python is one of the most popular programming languages. It is widely used in web development, data science, machine learning, and automation. Libraries such as NumPy, Pandas, and LangChain have made Python especially popular in the AI ecosystem.

Climate change refers to long-term shifts in global temperatures and weather patterns. Human activities, particularly the burning of fossil fuels, have increased greenhouse gas concentrations in the atmosphere. This has contributed to rising global temperatures and more frequent extreme weather events.
'''

from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from dotenv import load_dotenv

load_dotenv()

embedding_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=40
)

recursive_chunks = recursive_splitter.split_text(text)
print(f"recursive chunks : {len(recursive_chunks)}")
for i, chunk in enumerate(recursive_chunks) :
    print(f"\\n-- Chunk {i + 1} ({len(chunk)} chars) ---")
    print(chunk[:100] + " ... " if len(chunk) > 100 else chunk)

semantic_chunker = SemanticChunker(
    embedding_model,
    breakpoint_threshold_type='percentile',
    breakpoint_threshold_amount=70 # split at 90th percentile dissimilarity
)

semantic_chunks = semantic_chunker.split_text(text)
print(f"semantic chunks : {len(semantic_chunks)}")
for i, chunk in enumerate(semantic_chunks) :
    print(f"\\n-- Chunk {i + 1} ({len(chunk)} chars) ---")
    print(chunk[:100] + " ... " if len(chunk) > 100 else chunk)


recursive_vectorstore = Chroma.from_texts(
    recursive_chunks, embedding_model
)

semantic_vectorstore = Chroma.from_texts(
    semantic_chunks, embedding_model
)

test_queries = [
    "What is rest defence and why is it important?",
    "What engineering achievements were associated with the Roman Empire?",
    "How have fossil fuels contributed to climate change?"
]

for query in test_queries:

    rec_ret = recursive_vectorstore.similarity_search(
        query, k = 3
    )
    print(f"\n query : {query}\n\n")
    print("rec retrieved chunks : \n")

    for items in rec_ret :
        print(f"{items.page_content} \n")

    sem_ret = semantic_vectorstore.similarity_search(
        query, k = 3
    )

    print(f"query : {query}\n\n")
    print("sem retrieved chunks : \n")

    for items in sem_ret :
        print(f"{items.page_content} \n")
