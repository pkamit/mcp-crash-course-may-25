import asyncio
import os

import gradio as gr
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.prompts import ChatPromptTemplate
from agents import Agent, Runner, trace
from agents.mcp import MCPServerStdio
from playwright.async_api import async_playwright
import time

cdp_url = "ws://localhost:9000/devtools/browser/c454e791-291e-41f2-bb26-cbe1dd2e676a"


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

studio_server_parameters = StdioServerParameters(
    command="python",
    args=["/home/amit/Documents/gitprojects/mcp-crash-course/servers/math_server.py"],
)


async def fetch_tools():
    # async with async_playwright() as p:
    #     browser = await p.chromium.launch(headless=False, executable_path="/usr/bin/google-chrome")
    #     page = await browser.new_page()
    #     await page.goto("https://www.amazon.in")
    #     print(await page.title())
    #     await browser.close()

    playwright_params = {"command": "npx", "args": ["@playwright/mcp@latest"]}

    # async with MCPServerStdio(
    #     params=playwright_params, client_session_timeout_seconds=30
    # ) as server:
    #     playwright_tools = await server.list_tools()
    #     for tool in playwright_tools:
    #         print(f"{tool.name}")
    sandbox_path = os.path.abspath(os.path.join(os.getcwd(), "servers"))
    files_params = {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", sandbox_path],
    }

    # async with MCPServerStdio(
    #     params=files_params, client_session_timeout_seconds=30
    # ) as server:
    #     file_tools = await server.list_tools()
    #     for tool in file_tools:
    #         print(f"{tool.name}")
    # Updated instructions for price hunting
    price_instructions = """
    You browse e-commerce sites to find product prices. You:
    1. Navigate to major retailers (Amazon, Flipkart, etc.)
    2. Search for the specified product
    3. Accept cookies/dismiss popups if needed
    4. Locate the product's price element
    5. Note price and product details
    6. Try alternative sites if first attempt fails
    7. Always verify price matches the correct product
    """

    async with MCPServerStdio(
        params=files_params, client_session_timeout_seconds=30
    ) as mcp_server_files:
        async with MCPServerStdio(
            params=playwright_params, client_session_timeout_seconds=30
        ) as mcp_server_browser:
            agent = Agent(
                name="PriceHunter",
                instructions=price_instructions,  # Updated instructions
                model="gpt-4.1-mini",
                mcp_servers=[mcp_server_files, mcp_server_browser],
            )
            with trace("price_check"):
                # Modified query with price-focused parameters
                result = await Runner.run(
                    agent,
                    "Find the current price of PlayStation 5 on any e-commerce site. "
                    "Provide the details as markdown with product name, price, source URL, and date.",
                )

                print(result.final_output)
    output_path = os.path.join(sandbox_path, "ps5_prices.md")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result.final_output)
    print(f"Results saved to {output_path}")


async def agent_response(message, history):
    client = MultiServerMCPClient(
        {
            "math": {
                "command": "python",
                "args": [
                    "/home/amit/Documents/gitprojects/mcp-crash-course/servers/math_server.py"
                ],
                "transport": "stdio",
            },
            "weather": {
                "url": "http://localhost:8000/mcp",
                "transport": "streamable_http",
            },
        }
    )
    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.0, api_key=OPENAI_API_KEY)
    tools = await client.get_tools()
    agent = create_react_agent(llm, tools)
    # Add prompt as SystemMessage if needed
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=message),
    ]
    agent_response = await agent.ainvoke({"messages": messages})
    return agent_response["messages"][-1].content


# def gradio_chat(message, history):
#     return asyncio.run(agent_response(message, history))


# ui = gr.ChatInterface(gradio_chat)
# ui.launch()

if __name__ == "__main__":
    asyncio.run(fetch_tools())
