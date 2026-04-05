from typing import List
from pydantic import BaseModel, Field

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
load_dotenv()


class Source(BaseModel):
    """
    Schema for a source used by the agent.
    """
    url:str = Field(description="The url of the source")

class AgentResponse(BaseModel):
    """
    Schema for agent response with answer and sources.
    """
    answer: str = Field(description="The answer to the question")
    sources: List[Source] = Field(default_factory=list, description="List of sources used to generate the answer")
    

llm = ChatOpenAI()
tools = [TavilySearch()]
# agent = create_agent(model=llm, tools=tools)
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main():
    print("Hello from langchain-learn!")
    result = agent.invoke({"messages": [HumanMessage(content="What is the capital of China?")]})
    print(result)


if __name__ == "__main__":
    main()
