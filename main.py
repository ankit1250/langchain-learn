import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from tavily import TavilyClient
import tavily

load_dotenv()

tavily = TavilyClient()

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
    return "Useful data"

llm = ChatOpenAI(model="gpt-6-luna")
tools = [search,extractData]
agent = create_agent(model=llm,tools=tools)



def main():
    print("Hello from langchain-learn!")
    result = agent.invoke({"messages":HumanMessage("Search for 3 job postings for AI software engineer in India for Langchain and Langraph")})
    print(result)

if __name__ == "__main__":
    main()
