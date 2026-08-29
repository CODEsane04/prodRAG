"""
Langsmith setup & obervability
Production mnitoring for Langchian/Langragph
"""

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable
from langsmith.run_trees import RunTree
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGSMITH_TRACING"] = "true"

@traceable(name="basic_chaining")
def demo_basic_tracing(topic : str) -> str :
    """Basic langsmith tracing"""

    llm = ChatGoogleGenerativeAI(
        model="gemma-4-31b-it",
        temperature=0.1
    )

    prompt = PromptTemplate(
        template="explain this topic : {topic} in 2-3 lines only",
        input_variables=["topic"]
    )

    parser = StrOutputParser()

    chain = prompt | llm | parser
    result = chain.invoke({
        "topic" : topic
    })

    print(result)
    return result

if __name__ == "__main__" :
    demo_basic_tracing(topic="Counter pressing in context of football")
    