import os
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI
from elasticsearch import Elasticsearch
import gradio as gr
import re
import json

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL")
ES_INDEX = os.getenv("ES_INDEX", "products")

# Initialize clients
es = Elasticsearch(ELASTICSEARCH_URL)
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# --- DYNAMIC FIELD DISCOVERY ---
def extract_fields(mapping, prefix=""):
    """Recursively extract all fields from Elasticsearch mapping"""
    fields = []
    for field, props in mapping.items():
        field_type = props.get("type")
        if field_type == "object" or "properties" in props:
            subfields = extract_fields(
                props.get("properties", {}), prefix + field + "."
            )
            fields.extend(subfields)
        else:
            fields.append(prefix + field)
    return fields

def get_all_fields(index):
    """Get all available fields from Elasticsearch index"""
    try:
        mapping = es.indices.get_mapping(index=index)
        properties = mapping[index]["mappings"]["properties"]
        return extract_fields(properties)
    except Exception as e:
        print(f"Error getting fields: {e}")
        return ["name", "price", "sku", "category"]  # fallback

# Get available fields
ES_FIELDS = get_all_fields(ES_INDEX)
print("Discovered fields:", ES_FIELDS)

# --- FIELD ANALYSIS ---
def analyze_field_types():
    """Analyze field types to understand data structure better"""
    try:
        mapping = es.indices.get_mapping(index=ES_INDEX)
        properties = mapping[ES_INDEX]["mappings"]["properties"]
        
        field_info = {}
        for field, props in properties.items():
            field_type = props.get("type", "unknown")
            field_info[field] = field_type
            
        print("Field types discovered:", field_info)
        return field_info
    except Exception as e:
        print(f"Error analyzing field types: {e}")
        return {}

FIELD_TYPES = analyze_field_types()

# --- SAMPLE DATA ANALYSIS ---
def get_sample_data():
    """Get sample data to understand field values and structure"""
    try:
        sample_query = {
            "query": {"match_all": {}},
            "size": 5
        }
        response = es.search(index=ES_INDEX, body=sample_query)
        samples = [hit["_source"] for hit in response["hits"]["hits"]]
        print("Sample data structure:", samples[0] if samples else "No data")
        return samples
    except Exception as e:
        print(f"Error getting sample data: {e}")
        return []

# --- INTELLIGENT QUERY UNDERSTANDING ---
async def understand_user_intent(question):
    """Dynamically understand user intent and generate appropriate filters"""
    
    # Get sample data for context
    sample_data = get_sample_data()
    sample_context = ""
    if sample_data:
        sample_context = f"Sample product structure: {json.dumps(sample_data[0], indent=2)}"
    
    prompt = f"""
You are an intelligent product search assistant. Based on the user's question, extract:

1. **Search Filters**: Field-value pairs for filtering products
2. **Display Fields**: Which fields to show in results
3. **Query Type**: The type of search (exact match, range, fuzzy, etc.)

Available fields in the database: {', '.join(ES_FIELDS)}
Field types: {json.dumps(FIELD_TYPES, indent=2)}

{sample_context}

Instructions:
- For SKU searches: Use exact matching with "term" type
- For price ranges: Use numeric range queries with "range" type
- For categories/brands/names: ALWAYS use "match" type for case-insensitive search
- For product names: Use "match" type for text search
- If no specific fields requested for display, show: name, price, sku, category
- Extract all relevant filter conditions from the question
- For text fields like category, brand, name - always use "match" type, never "term"

Return JSON format:
{{
    "filters": [
        {{"field": "field_name", "value": "field_value", "type": "match|term|range|terms"}}
    ],
    "display_fields": ["field1", "field2", ...],
    "query_intent": "description of what user is looking for"
}}

Examples of correct query types:
- "show laptops under $500" → {{"field": "price", "value": 500, "type": "range"}}, {{"field": "name", "value": "laptop", "type": "match"}}
- "find SKU ABC123" → {{"field": "sku", "value": "ABC123", "type": "term"}}
- "apple products in electronics" → {{"field": "brand", "value": "apple", "type": "match"}}, {{"field": "category", "value": "electronics", "type": "match"}}
- "products with 8GB RAM" → {{"field": "specification.ram", "value": "8GB", "type": "match"}}

IMPORTANT: Always use "match" type for category, brand, name fields to handle case-insensitive search!

User Question: "{question}"
"""

    try:
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0,
        )
        
        content = response.choices[0].message.content.strip()
        print(f"LLM Intent Analysis: {content}")
        
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group(0))
            return result
        else:
            raise ValueError("No JSON found in response")
            
    except Exception as e:
        print(f"Error in intent understanding: {e}")
        # Fallback to basic pattern matching
        return fallback_intent_analysis(question)

def fallback_intent_analysis(question):
    """Fallback method for intent analysis using regex patterns"""
    filters = []
    display_fields = ["name", "price", "sku", "category"]
    
    # SKU pattern matching
    sku_pattern = r'\b[A-Z]{2,}[0-9]{3,}\b'
    skus = re.findall(sku_pattern, question.upper())
    if skus:
        if len(skus) == 1:
            filters.append({"field": "sku", "value": skus[0], "type": "term"})
        else:
            filters.append({"field": "sku", "value": skus, "type": "terms"})
    
    # Price range patterns
    price_patterns = [
        (r'under \$?(\d+)', 'lt'),
        (r'below \$?(\d+)', 'lt'),
        (r'above \$?(\d+)', 'gt'),
        (r'over \$?(\d+)', 'gt'),
    ]
    
    for pattern, operator in price_patterns:
        match = re.search(pattern, question.lower())
        if match:
            price = float(match.group(1))
            filters.append({"field": "price", "value": price, "type": "range", "operator": operator})
    
    # Category/brand matching - use case-insensitive match
    keywords = question.lower().split()
    common_categories = ['electronics', 'laptops', 'phones', 'tablets', 'accessories', 'electronic']
    for category in common_categories:
        if category in keywords:
            # Use match type for case-insensitive search
            filters.append({"field": "category", "value": category, "type": "match"})
    
    # Brand detection
    common_brands = ['apple', 'samsung', 'sony', 'hp', 'dell', 'lenovo']
    for brand in common_brands:
        if brand in keywords:
            filters.append({"field": "brand", "value": brand, "type": "match"})
    
    return {
        "filters": filters,
        "display_fields": display_fields,
        "query_intent": f"Search for: {question}"
    }

# --- DYNAMIC ELASTICSEARCH QUERY BUILDER ---
def build_dynamic_es_query(intent_result):
    """Build Elasticsearch query based on intelligent intent analysis"""
    
    filters = intent_result.get("filters", [])
    
    if not filters:
        return {"query": {"match_all": {}}}
    
    must_clauses = []
    
    for filter_item in filters:
        field = filter_item["field"]
        value = filter_item["value"]
        query_type = filter_item["type"]
        
        # Case-insensitive handling for text fields
        if query_type == "term":
            # Use match with case-insensitive search for text fields like category
            if field in ["category", "brand", "name"] or "category" in field.lower():
                must_clauses.append({"match": {field: {"query": value, "operator": "and"}}})
            else:
                must_clauses.append({"term": {field: value}})
        elif query_type == "terms":
            # For multiple values, use should clause with match for case-insensitive
            if field in ["category", "brand", "name"] or "category" in field.lower():
                should_clauses = []
                for val in value:
                    should_clauses.append({"match": {field: {"query": val, "operator": "and"}}})
                must_clauses.append({"bool": {"should": should_clauses, "minimum_should_match": 1}})
            else:
                must_clauses.append({"terms": {field: value}})
        elif query_type == "match":
            # Use case-insensitive match
            must_clauses.append({"match": {field: {"query": value, "operator": "and"}}})
        elif query_type == "range":
            operator = filter_item.get("operator", "gte")
            must_clauses.append({"range": {field: {operator: value}}})
        elif query_type == "fuzzy":
            must_clauses.append({"fuzzy": {field: {"value": value, "fuzziness": "AUTO"}}})
    
    if len(must_clauses) == 1:
        query = {"query": must_clauses[0]}
    else:
        query = {"query": {"bool": {"must": must_clauses}}}
    
    print(f"Generated ES Query: {json.dumps(query, indent=2)}")
    return query

# --- SEARCH EXECUTION ---
async def execute_search(intent_result):
    """Execute search based on intent analysis"""
    
    es_query = build_dynamic_es_query(intent_result)
    display_fields = intent_result.get("display_fields", ["name", "price", "sku", "category"])
    
    try:
        response = es.search(index=ES_INDEX, body=es_query, size=50)
        hits = response["hits"]["hits"]
        
        print(f"Found {len(hits)} products")
        
        if not hits:
            return "No products found matching your criteria.", []
        
        # Format results
        products = []
        for hit in hits:
            product_data = hit["_source"]
            formatted_product = {}
            
            for field in display_fields:
                value = get_nested_value(product_data, field)
                if value is not None:
                    formatted_product[field] = value
            
            products.append(formatted_product)
        
        return None, products
        
    except Exception as e:
        print(f"Search execution error: {e}")
        return f"Search error: {str(e)}", []

def get_nested_value(data, field_path):
    """Get value from nested dictionary using dot notation"""
    keys = field_path.split(".")
    current = data
    
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return None
    
    return current

# --- DYNAMIC RESPONSE GENERATION ---
async def generate_dynamic_response(question, products, intent_result):
    """Generate natural language response based on found products"""
    
    if not products:
        return "I couldn't find any products matching your criteria. Please try a different search."
    
    # Create product summary
    product_summary = []
    for product in products[:10]:  # Limit to first 10 for response
        product_line = []
        for field, value in product.items():
            product_line.append(f"{field}: {value}")
        product_summary.append("- " + ", ".join(product_line))
    
    products_text = "\n".join(product_summary)
    query_intent = intent_result.get("query_intent", "product search")
    
    prompt = f"""
Based on the user's question and the found products, provide a helpful and natural response.

User's Question: "{question}"
Query Intent: {query_intent}
Number of products found: {len(products)}

Products Found:
{products_text}

Instructions:
- Provide a conversational and helpful response
- Include relevant product details that answer the user's question
- If many products found, summarize key information
- Be specific about product attributes mentioned
- If it's a SKU availability question, clearly state if products are available
- Format the response in a user-friendly way

Response:"""

    try:
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.3,
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"Response generation error: {e}")
        # Fallback to simple formatting
        return format_simple_response(question, products)

def format_simple_response(question, products):
    """Simple fallback response formatting"""
    if not products:
        return "No products found."
    
    response_lines = [f"Found {len(products)} product(s):"]
    
    for i, product in enumerate(products[:5], 1):
        line_parts = []
        for field, value in product.items():
            line_parts.append(f"{field}: {value}")
        response_lines.append(f"{i}. {', '.join(line_parts)}")
    
    if len(products) > 5:
        response_lines.append(f"... and {len(products) - 5} more products.")
    
    return "\n".join(response_lines)

# --- MAIN QUESTION PROCESSING ---
async def process_question(question):
    """Main function to process user questions dynamically"""
    
    print(f"\n=== Processing Question: {question} ===")
    
    try:
        # Step 1: Understand user intent
        intent_result = await understand_user_intent(question)
        print(f"Intent Analysis: {json.dumps(intent_result, indent=2)}")
        
        # Step 2: Execute search
        error, products = await execute_search(intent_result)
        
        if error:
            return error
        
        # Step 3: Generate response
        response = await generate_dynamic_response(question, products, intent_result)
        
        return response
        
    except Exception as e:
        print(f"Error processing question: {e}")
        return f"I encountered an error while processing your question: {str(e)}"

# --- GRADIO INTERFACE ---
def sync_process_question(question):
    """Synchronous wrapper for Gradio"""
    return asyncio.run(process_question(question))

# Test function
def test_system():
    """Test the system with sample questions"""
    test_questions = [
        "show me laptops under $500",
        "are SKU ABC123 and XYZ789 available?",
        "find apple products in electronics",
        "what tablets do you have with 64GB storage?",
        "show me all products with their prices"
    ]
    
    print("\n=== Testing System ===")
    for question in test_questions:
        print(f"\nQ: {question}")
        try:
            result = asyncio.run(process_question(question))
            print(f"A: {result}")
        except Exception as e:
            print(f"Error: {e}")

# Gradio Interface
with gr.Blocks(title="Dynamic Product Q&A System") as demo:
    gr.Markdown("# 🔍 Dynamic Product Q&A System")
    gr.Markdown("Ask any question about products - the system will intelligently understand your query and search accordingly!")
    
    with gr.Row():
        with gr.Column():
            question_input = gr.Textbox(
                label="Ask about products",
                placeholder="e.g., 'show laptops under $500', 'is SKU ABC123 available?', 'find apple phones'",
                lines=2
            )
            ask_button = gr.Button("🔍 Search", variant="primary")
        
        with gr.Column():
            answer_output = gr.Textbox(
                label="Answer",
                lines=10,
                show_copy_button=True
            )
    
    gr.Examples(
        examples=[
            ["show me laptops under $500"],
            ["are these SKUs available: ABC123, XYZ789"],
            ["find apple products in electronics category"],
            ["what tablets have 64GB storage?"],
            ["show all products with price and category"],
            ["products above $1000 in electronics"]
        ],
        inputs=question_input
    )
    
    ask_button.click(
        fn=sync_process_question,
        inputs=question_input,
        outputs=answer_output
    )

if __name__ == "__main__":
    # Test the system first
    test_system()
    
    # Launch Gradio interface
    demo.launch(share=True)