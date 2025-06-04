import os
import gradio as gr
import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI

# Load environment variable
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Prepare OpenAI async client
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Load and preprocess CSV
PRODUCTS_CSV = "flipkart_products.csv"
df = pd.read_csv(PRODUCTS_CSV)
df.columns = df.columns.str.strip()

def extract_product_id(url):
    try:
        return url.split('/')[-1]
    except Exception:
        return None

df['product_id'] = df['url'].apply(extract_product_id)

def lookup_product(product_id):
    row = df[df['product_id'] == product_id]
    if row.empty:
        return None
    row = row.iloc[0]
    return {
        "name": row["name"],
        "price": row.get("price", ""),
        "highlights": row["highlights"]
    }

async def format_with_gpt35(product_info):
    prompt = (
        "Format the following product info as markdown:\n\n"
        f"Name: {product_info['name']}\n"
        f"Price: {product_info['price']}\n"
        f"Highlights: {product_info['highlights']}\n\n"
        "Output: The name as a heading, price in the next row, and highlights as a bullet list. No extra commentary."
    )
    response = await client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()

async def chat_fn(message, history):
    product_id = message.strip()
    product_info = lookup_product(product_id)
    if not product_info:
        return f"❌ Product ID `{product_id}` not found."
    markdown = await format_with_gpt35(product_info)
    return markdown

demo = gr.ChatInterface(
    fn=chat_fn,
    title="Flipkart Product Info Chat",
    description="Enter a product ID (e.g., itma403c7d655267) to get product highlights.",
    type="messages"
)

if __name__ == "__main__":
    demo.launch()
