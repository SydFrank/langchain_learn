from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import HumanMessage
from langchain_openai import ChatOpenAI
from tavily import TavilyClient
import os


load_dotenv()
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def search(query: str) -> str:
    """
    Tool that searches over internet

    Args:
        query: The query to search for

    Returns:
        The search results
    """
    print(f"Searching the web for: {query}")
    return tavily.search(query=query)



llm = ChatOpenAI()
tools = [search]
agent = create_agent(model=llm, tools=tools)


def main():
    print("Hello from langchain-learn!")
    result = agent.invoke({"messages": [HumanMessage(content="What is the capital of China?")]})
    print(result)


if __name__ == "__main__":
    main()
