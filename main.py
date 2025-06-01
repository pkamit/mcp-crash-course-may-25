import asyncio
import os

import gradio as gr
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage , SystemMessage
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

studio_server_parameters = StdioServerParameters(
    command="python",
    args=["/home/amit/Documents/gitprojects/mcp-crash-course/servers/math_server.py"],
)


async def agent_response(message, history):
    client = MultiServerMCPClient({
        "math": {
            "command": "python",
            "args": ["/home/amit/Documents/gitprojects/mcp-crash-course/servers/math_server.py"],
            "transport": "stdio",
        },
        "weather": {
            "url": "http://localhost:8000/mcp",
            "transport": "streamable_http",
        }
    })
    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.0, api_key=OPENAI_API_KEY)
    tools = await client.get_tools()
    agent = create_react_agent(llm, tools)
    # Add prompt as SystemMessage if needed
    messages = [SystemMessage(content="You are a helpful assistant."), HumanMessage(content=message)]
    agent_response = await agent.ainvoke({"messages": messages})
    return agent_response["messages"][-1].content



def gradio_chat(message, history):
    return asyncio.run(agent_response(message, history))


ui = gr.ChatInterface(gradio_chat)
ui.launch()
