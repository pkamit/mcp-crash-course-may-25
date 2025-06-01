import asyncio
import os

from dotenv import load_dotenv
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from mcp import ClientSession, StdioServerParameters

from mcp.client.stdio import stdio_client

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI()


studio_server_parameters = StdioServerParameters(
    command="python",
    args=["/home/amit/Documents/gitprojects/mcp-crash-course/servers/math_server.py"],
)

stdio_client = stdio_client(studio_server_parameters)

async def main():
    print("Hello from mcp-crash-course!")


if __name__ == "__main__":
    asyncio.run(main()) 
