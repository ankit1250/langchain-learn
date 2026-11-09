import os

from typing import List

from pydantic import BaseModel, Field

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from tavily import TavilyClient
import tavily
from langchain_tavily import TavilySearch

load_dotenv()

tavily = TavilyClient()

class Source(BaseModel):
    """Schema for the source used by the agent"""
    
    url:str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""

    answer:str = Field(description="The answer to the user's question")
    sources:List[Source] = Field(default_factory=list, description="The sources used to answer the question")


@tool
def search(query: str) -> str:
    """
    Tool that searches over internet

    Args:
        query: the string to search for 
    
    Returns:
        The search result
    """

    print(f"Searching for {query}")
    return tavily.search(query=query)

@tool
def extractData(rawdata:str) -> str:
    """
    Extracts useful data from raw unfiltered data

    Args:
        rawdata: any raw string or paragraph
    Returns:
        The extracted data
    """
    return tavily.extract(rawdata)

llm = ChatOpenAI(model="gpt-5")
tools = [search]
agent = create_agent(model=llm,tools=tools,response_format=AgentResponse)



def main():
    print("Hello from langchain-learn!")
    result = agent.invoke({"messages":HumanMessage("Search for 3 job postings for AI software engineer in India for Langchain and Langraph")})
    print(result)

if __name__ == "__main__":
    main()
