import os
import tempfile
from langchain_community.document_loaders import TextLoader, PyPDFLoader

def load_text_file():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
        temp_file.write(b"hello this is a simple text file")
        temp_file_path = temp_file.name

    try:
        loader = TextLoader(temp_file_path)
        documents = loader.load()

        for doc in documents:
            print(doc.page_content)

    finally:
        os.remove(temp_file_path)

def pdf_loader(pdf_path : str) :

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    print(f"loaded {len(documents)} documents from PDF")
    for i, doc in enumerate(documents) :
        print(f"document {i + 1} content previre : {doc.page_content[:100]}")
        print(f"metadata : {doc.metadata}")


if __name__ == "__main__":

    pdf_path = "/Users/debjitghorai/prod_rag/sample.pdf"
    pdf_loader(pdf_path)
    #load_text_file()