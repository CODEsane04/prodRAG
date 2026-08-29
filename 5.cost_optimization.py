
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()


class TokenBudget:

    def __init__(self, max_tokens_per_request: int = 4000):
        self.max_per_request = max_tokens_per_request
        self.usage = {
            "total_input": 0,
            "total_output" : 0,
            "requests" : 0
        }

    def estimate_tokens(self, text : str) -> int :
        """rough token estimation (actual would use tiktokens)"""
        return int(len(text.split()) * 1.3)

    def check_budget(self, text : str) -> tuple[bool, int]:
        """check fi requests are in the budget"""
        tokens = self.estimate_tokens(text)

        if tokens <= self.max_per_request :
            return True,tokens
        else :
            return False,tokens
       

    def record_usage(self, input_tokens : int, output_tokens : int) :
        """record token usage"""
        self.usage["total_input"] += input_tokens
        self.usage["total_output"] += output_tokens
        self.usage["requests"] += 1

    def get_stats(self) -> dict :
        print(self.usage)


class BudgetedLLM :
    """LLM with token budgeting"""

    def __init__(self, max_tokens : int = 4000):
        self.llm = ChatGoogleGenerativeAI(
            model="gemma-4-31b-it",
            temperature=0.1
        )
        self.budget = TokenBudget(max_tokens_per_request=max_tokens)

    #@traceable(name="bugeted_invoke")
    def invoke(self, query: str) -> str :

        #check_budget
        within_budget, tokens = self.budget.check_budget(query)

        if not within_budget :
            return f"query exceeds token budget  : {tokens} > {self.budget.max_per_request}"

            
        

        #execute
        parser = StrOutputParser()
        chain = self.llm | parser
        response = chain.invoke(query)

        #record usage
        output_tokens = self.budget.estimate_tokens(response)
        self.budget.record_usage(tokens, output_tokens)

        return response

    def get_stats(self) -> dict :
        return self.budget.get_stats()


def demo_token_budgeting() : 
    """demonstrate token budgeting"""

    llm = BudgetedLLM(max_tokens=100)
    queries = [
        "what is AI? answer in 2 lines only",
        " explain" + (" very " * 100) + "complex topic "
    ]

    print("token budgeting demo\n")

    for query in queries :
        print(f"trying to execute. query : {query}\n")
        response = llm.invoke(query)

        print(response)
        print("\n")
        llm.get_stats()

if __name__ == "__main__" :
    demo_token_budgeting()