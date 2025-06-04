import os
import re
import pandas as pd
from dotenv import load_dotenv
import gradio as gr
import chromadb
from chromadb.utils import embedding_functions
from openai import AsyncOpenAI

# --- Load environment and API keys ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# --- Load and preprocess CSV ---
PRODUCTS_CSV = "flipkart_products.csv"
df = pd.read_csv(PRODUCTS_CSV)
df.columns = df.columns.str.strip()

def extract_product_id(url):
    return url.split("/")[-1]

df['product_id'] = df['url'].apply(extract_product_id)
df['chunk'] = df.apply(
    lambda row: f"Name: {row['name']}\nPrice: {row['price']}\nHighlights: {row['highlights']}", axis=1
)

# --- Setup Chroma vector DB ---
chroma_client = chromadb.Client()
embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY, model_name="text-embedding-3-small"
)
if "products" in [c.name for c in chroma_client.list_collections()]:
    chroma_client.delete_collection("products")
collection = chroma_client.create_collection("products", embedding_function=embedding_fn)
for idx, row in df.iterrows():
    collection.add(
        documents=[row['chunk']],
        ids=[row['product_id']]
    )

# --- Hybrid retrieval function ---
def is_product_id(query):
    # Adjust regex to match your product ID format
    return bool(re.match(r"^itm[a-z0-9]{10,}$", query.strip()))

async def hybrid_chat_fn(message, history):
    query = message.strip()
    if is_product_id(query):
        # Direct lookup
        row = df[df['product_id'] == query]
        if row.empty:
            return f"❌ Product ID `{query}` not found."
        row = row.iloc[0]
        prompt = (
            "Format the following product info as markdown:\n\n"
            f"Name: {row['name']}\n"
            f"Price: {row['price']}\n"
            f"Highlights: {row['highlights']}\n\n"
            "Output: The name as a heading, price in the next row, and highlights as a bullet list. No extra commentary."
        )
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    else:
        # Semantic RAG for natural language queries
        results = collection.query(query_texts=[query], n_results=3)
        if not results['documents'][0]:
            return "No relevant product found."
        context = "\n\n".join(results['documents'][0])
        prompt = (
            f"You are a helpful assistant. Use the following product information to answer the user's question.\n\n"
            f"Context:\n{context}\n\n"
            f"User Query: {query}\n\n"
            "Present the answer in markdown format with product name as heading, price in the next row, and highlights as a bullet list."
        )
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()

# --- Gradio Chat Interface ---
demo = gr.ChatInterface(
    fn=hybrid_chat_fn,
    title="Flipkart Product Info RAG Chat",
    description="Ask about a product by ID (e.g., itma403c7d655267) or by description (e.g., '8GB RAM phone').",
    type="messages"
)

if __name__ == "__main__":
    demo.launch()
