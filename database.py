from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain.callbacks.tracers import LangChainTracer
from langchain_core.callbacks import CallbackManager
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

import time
import os

import pandas as pd
import re
import ast
import csv
import requests

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
    temperature = 0.3,
    model_name = 'llama3-70b-8192',
    callback_manager = CallbackManager([tracer])
)


# add data to pinecone
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

# fetch similar titles
def fetch_similar_titles(query, vector_store=vector_store, threshold=0.65):
    result = vector_store.similarity_search_with_score(query=query, k=10)
    title_ids = [(title.id) for title, score in  result if score > 0.40]
    return title_ids
    

# generate SQL query
def generate_sql_query(ids):
    write_query = create_sql_query_chain(llm, db)
    execute_query = QuerySQLDataBaseTool(db=db)

    question = (
        f"Generate a SQL query to retrieve all columns for the given primary keys: {ids}. "
        "The 'index' column is the primary key. "
        "Only return the SQL query, nothing else."
        "Do not use LIMIT return all the records"
    )

    response = write_query.invoke({"question": question})

    if isinstance(response, dict) and "query" in response:
        sql_query = response["query"]
    else:
        match = re.search(r"SQLQuery:\s*(SELECT .*);?", response, re.DOTALL | re.IGNORECASE)
        if match:
            sql_query = match.group(1).strip()
        else:
            raise ValueError("SQL query could not be extracted.")

    query_result = execute_query.invoke(sql_query)
    output = ast.literal_eval(query_result)

    columns = ['index', 'category', 'title', 'description', 'url']

    output_df = pd.DataFrame(output, columns=columns)
    return output_df



