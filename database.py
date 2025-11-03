from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq
from langchain_core.tracers.langchain import LangChainTracer
from langchain_core.callbacks import CallbackManager
from dotenv import load_dotenv
load_dotenv()
import time
import os
import pandas as pd
import re
import ast

os.environ['LANGSMITH_TRACING']= os.getenv('LANGSMITH_TRACING')
os.environ['LANGSMITH_ENDPOINT']= os.getenv('LANGSMITH_ENDPOINT')
os.environ['LANGSMITH_API_KEY']= os.getenv('LANGSMITH_API_KEY')
os.environ['LANGSMITH_PROJECT']= os.getenv('LANGSMITH_PROJECT')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')


tracer = LangChainTracer()
callback_manager = CallbackManager([tracer])

pc = Pinecone(api_key= PINECONE_API_KEY)
index_name = 'chatbot-titles-index'

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name = index_name,
        dimension = 1536,
        metric = 'cosine',
        spec= ServerlessSpec(
            cloud = 'aws',
            region= 'us-east-1'
        )
    )
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

print('Index is Ready')

# connecting to SQL database
index  = pc.Index(index_name)
vector_store = PineconeVectorStore(index = index, embedding=OpenAIEmbeddings(model='text-embedding-3-small'))
db = SQLDatabase.from_uri(f'sqlite:///all_categories_data-sqlite.db')

# initalising an LLM
llm = ChatGroq(
    temperature = 0.1,
    model_name = 'llama-3.1-8b-instant',
    callback_manager = CallbackManager([tracer])
)

# ============================================
# SECURITY VALIDATION FUNCTIONS
# ============================================

def validate_query(query):
    """Validate and sanitize user input"""
    if not query or not isinstance(query, str):
        raise ValueError("Query must be a non-empty string")
    
    query = query.strip()
    
    if len(query) > 500:
        raise ValueError("Query exceeds maximum length")
    
    dangerous_patterns = [
        r'<script', r'javascript:', r'<iframe', r'DROP\s+TABLE', 
        r'DELETE\s+FROM', r'INSERT\s+INTO', r'UPDATE\s+.*\s+SET',
        r'--', r'UNION\s+SELECT', r'EXEC\s*\('
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            raise ValueError("Invalid input detected")
    
    return query


def validate_ids(ids):
    """Validate IDs are positive integers"""
    if not ids or not isinstance(ids, list):
        raise ValueError("IDs must be a non-empty list")
    
    validated_ids = []
    for id_val in ids:
        try:
            id_int = int(id_val)
            if id_int <= 0:
                raise ValueError(f"ID must be positive: {id_val}")
            validated_ids.append(id_int)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid ID: {id_val}")
    
    return validated_ids

# ============================================
# MAIN FUNCTIONS
# ============================================

def add_data():
    df = pd.read_excel('all_categories_data.xlsx')
    all_titles = df['title']
    document_list = []
    ids_list = []
    for idx, title in enumerate(all_titles):
        print(f'Adding record count: {idx}')
        document = Document(page_content = title)
        document_list.append(document)
        ids_list.append(str(idx + 1))
    vector_store.add_documents(documents=document_list, ids=ids_list)
    print('Data Added Successfully')

# add_data()


def fetch_similar_titles(query, vector_store=vector_store, threshold=0.65):
    """Fetch similar book titles from vector store"""
    try:
        query = validate_query(query)
    except ValueError as e:
        print(f"Invalid query: {e}")
        return []
    
    result = vector_store.similarity_search_with_score(query=query, k=20)
    
    title_ids = []
    for title, score in result:
        if score > 0.40:
            try:
                title_ids.append(int(title.id))
            except (ValueError, TypeError):
                print(f"Warning: Skipping invalid ID: {title.id}")
                continue
    
    print('title Ids', title_ids)
    return title_ids


def generate_sql_query(ids):
    """
    Query database directly - NO LLM NEEDED
    Returns DataFrame grouped by category
    """
    try:
        ids = validate_ids(ids)
    except ValueError as e:
        print(f"Invalid IDs: {e}")
        return pd.DataFrame()
    
    if not ids:
        print("No valid IDs provided")
        return pd.DataFrame()
    
    # DIRECT SQL QUERY - No LLM, no parsing errors
    ids_str = ','.join(str(id) for id in ids)
    sql_query = f'SELECT "index", title, category, description, url FROM Categories WHERE "index" IN ({ids_str}) ORDER BY category, title'
    
    print('SQL query:', sql_query)
    
    try:
        # Execute query directly
        query_result = db.run(sql_query)
        
        # Parse results
        output = ast.literal_eval(query_result)
        
        # Create DataFrame with clear column names
        df = pd.DataFrame(output, columns=['id', 'title', 'category', 'description', 'url'])
        
        print(f"\nFound {len(df)} books in {df['category'].nunique()} categories")
        
        # Display grouped by category
        print("\n" + "="*60)
        for category in df['category'].unique():
            category_books = df[df['category'] == category]
            print(f"\n📚 {category} ({len(category_books)} books)")
            print("-" * 60)
            for idx, row in category_books.iterrows():
                print(f"  • {row['title']}")
                if pd.notna(row['url']):
                    print(f"    🔗 {row['url']}")
        print("="*60)
        
        return df
        
    except Exception as e:
        print(f"Error executing query: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()
