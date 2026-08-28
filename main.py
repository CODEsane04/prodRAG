from dotenv import load_dotenv
load_dotenv()

from langchain_core import __version__ as core_version
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

print(f"langchain-core verison : {core_version}")


def main():
    print("hello from prof rag")
    model=ChatGoogleGenerativeAI(
        model="gemma-4-31b-it",
        temperature=0.0
    )

    parser = StrOutputParser()

    chain = model | parser
    response = chain.invoke("only output what is asked, donot ever output your chan-of-thoughts or thinking process. question - what do you know abut the movie the interstellar?")

    print(response)

if __name__ == "__main__":
    main()