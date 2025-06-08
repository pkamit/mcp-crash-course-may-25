import os
import re
import glob
import pandas as pd
from dotenv import load_dotenv
import gradio as gr
import chromadb
from chromadb.utils import embedding_functions
from openai import AsyncOpenAI
from agents import Agent, Runner, trace

# --- Load environment and API keys ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = AsyncOpenAI(api_key=OPENAI_API_KEY)
# Add to environment variables
GRADIO_USER = os.getenv("GRADIO_USERNAME")
GRADIO_PASS = os.getenv("GRADIO_PASSWORD")

# --- Load and preprocess all CSVs ---
csv_files = glob.glob("*_products.csv")
dfs = []
for file in csv_files:
    source = file.replace("_products.csv", "")
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()
    df["source"] = source
    # Extract product_id from URL
    df["product_id"] = df["url"].apply(lambda url: url.split("/")[-1])
    df["chunk"] = df.apply(
        lambda row: f"Name: {row['name']}\nPrice: {row['price']}\nHighlights: {row['highlights']}",
        axis=1,
    )
    dfs.append(df)
df_all = pd.concat(dfs, ignore_index=True)

# --- Setup Chroma vector DB (single collection) ---
chroma_client = chromadb.Client()
embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY, model_name="text-embedding-3-small"
)
if "all_products" in [c.name for c in chroma_client.list_collections()]:
    chroma_client.delete_collection("all_products")
collection = chroma_client.create_collection(
    "all_products", embedding_function=embedding_fn
)

for idx, row in df_all.iterrows():
    metadata = {"source": row["source"]}
    collection.add(
        documents=[row["chunk"]], ids=[row["product_id"]], metadatas=[metadata]
    )


# --- Hybrid retrieval function ---
def is_product_id(query):
    # Adjust regex to match your product ID format
    return bool(re.match(r"^itm[a-z0-9]{10,}$", query.strip()))


async def hybrid_chat_fn(message, history):
    query = message.strip()
    # Optional: let user specify source, e.g., "flipkart:itma403c7d655267" or "amazon:itmx123456789"
    if ":" in query:
        source, pid = query.split(":", 1)
        source = source.strip().lower()
        pid = pid.strip()
    else:
        source, pid = None, query

    if is_product_id(pid):
        # Direct lookup with optional source filtering
        if source:
            row = df_all[(df_all["product_id"] == pid) & (df_all["source"] == source)]
        else:
            row = df_all[df_all["product_id"] == pid]
        if row.empty:
            return f"❌ Product ID `{pid}` not found."
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
        # Semantic RAG for natural language queries, with optional source filtering
        where = {"source": source} if source else None
        results = collection.query(query_texts=[query], n_results=3, where=where)
        if not results["documents"][0]:
            return "No relevant product found."
        context = "\n\n".join(results["documents"][0])
        prompt = (
            f"You are a helpful assistant. Use the following product information to answer the user's question.\n\n"
            f"Context:\n{context}\n\n"
            f"User Query: {query}\n\n"
            "Present the answer in markdown format with product name as heading, price in the next row, and highlights as a bullet list."
        )
        with trace("My Traced Chat Completion"):
            response = await client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.2,
            )
            print("Response:", response.choices[0].message.content.strip())
            print("Prompt tokens used:", response.usage.prompt_tokens)
            print("Completion tokens used:", response.usage.completion_tokens)
            print("Total tokens used:", response.usage.total_tokens)
            return response.choices[0].message.content.strip()


# --- Gradio Chat Interface ---
demo = gr.ChatInterface(
    fn=hybrid_chat_fn,
    title="Multi-Source Product Info RAG Chat",
    description="Ask about a product by ID (e.g., itma403c7d655267), by description (e.g., '8GB RAM phone'), or specify a source (e.g., 'flipkart:itma403c7d655267').",
    type="messages",
)

if __name__ == "__main__":
    # Modify launch command
    demo.launch(
        auth=(GRADIO_USER, GRADIO_PASS),
        auth_message="🔒 Enter credentials to access product search",
        rate_limit=[10, 300]
    )
