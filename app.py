import os
import json
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from database import *
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.prompts import PromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.graph.message import add_messages 
from typing_extensions import TypedDict

app = Flask(__name__)

CORS(app, origins= "*" )

app.config['CORS_HEADERS'] = 'Content-Type'

memory = MemorySaver()
thread_id = 1

class State(TypedDict):
    intent: str
    user_query: Annotated[list, add_messages]
    context: Annotated[list, add_messages]
    response: Annotated[list, add_messages]

graph_builder=StateGraph(State)
def answer_query(state:State):
    if state['intent'] == "Greeting":
        print("Debug: Greeting detected, skipping SQL query.")
        return {'context': ''}  
    
    elif state['intent'] == "Query":
        query = state['user_query'][-1]
        print('Debug: Query:', query.content)

        similar_title_ids = fetch_similar_titles(vector_store=vector_store, query=query.content) or []

        if similar_title_ids:
            sql_query_result = generate_sql_query(ids=similar_title_ids)
            context = sql_query_result.to_string() if isinstance(sql_query_result, pd.DataFrame) else sql_query_result
        else:
            print("Debug: No similar titles found.")
            context = ''

        return {'context': context}
    
    else:
        print("Debug: Unknown intent, returning empty context.")
        return {'context': ''}  

def identify_intent(state:State):
    prompt = f"""
    You are an intent classification model. Classify the following query into one of the following intents:
    When a user wishes to explore a category the intent should be query.

    1. Greeting
    2. Query
    3. Unknown
    
    Query: "{state['user_query']}"
    
    Respond with only the intent name.
    """

    response = llm.invoke(prompt)
    intent = response.content.strip()
    
    return {'intent': intent}

def greeting_answer(state:State):
    qa_prompt = PromptTemplate.from_template(
      """ Your name is Bharti. You are an AI assistant for the Indian Culture Portal that deal with Indian Culture and History.
           When a greets you you should reply with a formal greeting.

           From time to time you can a quirky response as well!

           You personality is of a smartass and know it all.

           Add emojis wherever necessary. But not much of it.
        """
    )
    if state['intent'] == "Greeting":
        chain = qa_prompt | llm
        response = chain.invoke({'question': state['user_query']})
        return {'response': response.content}


def query_answer(state:State):
    qa_prompt = PromptTemplate.from_template(
      """ You are an AI assistant for the Indian Culture Portal, specializing in Indian culture, history, and governance.

            **Instructions:**
            - Answer ONLY based on the provided context.
            - DO NOT make up information or provide random guesses.
            - If related topics are available, suggest them in a structured manner.
            - Do not provide personal opinions or views.
            - If URLs are NA do not include them in your answer
            - When you answer with ebooks, rarebooks, archives always mention their urls.
            - If description is available present it in a sumamrise manner.
            - Always answer from the context available do not go beyond it.
            
            **Response Formatting:**
            - Use **bold headings** for clarity.
            - Use bullet points for listing relevant books, laws, or documents.
            - Keep the answer concise and informative.
            - Use emojis **sparingly** to enhance readability (not excessively).

           context: {context}
           User question: {question}   
        """
    )
    if state['intent'] == "Query":

        chain = qa_prompt | llm
        response = chain.invoke({'context': state['context'], 'question': state['user_query']})
        return {'response': response.content}


graph_builder.add_node("question_type", identify_intent)
graph_builder.add_node("context_memory", answer_query)
graph_builder.add_node("greeting_response", greeting_answer)
graph_builder.add_node("query_response", query_answer)


graph_builder.add_edge(START,"question_type")
graph_builder.add_edge("question_type", "context_memory")
graph_builder.add_edge("context_memory", 'greeting_response')
graph_builder.add_edge("context_memory", 'query_response')
graph_builder.add_edge("query_response", END)
graph_builder.add_edge("greeting_response", END)

graph = graph_builder.compile(checkpointer=memory)


@app.post('/chat')
def query():
    events_list = []
    answer = ''

    data = request.get_json()
    user_query = data['query']

    config = {"configurable": {"thread_id": thread_id}}
    events = graph.stream(
        {'user_query': [{'role': 'user', 'content': user_query}]},
        config = config
    )
    
    for event in events:
        events_list.append(event)


    print(events_list[0]['question_type']['intent'])
    print(events_list)

    if events_list[0]['question_type']['intent'] == 'Query':
        answer = events_list[2].get('query_response', '') or events_list[3].get('greeting_response', '')
        answer = events_list[3].get('query_response', '') or events_list[2].get('greeting_response', '')
        return jsonify({'answer': answer['response']}), 200
    
    elif events_list[0]['question_type']['intent'] == 'Greeting':
        answer = events_list[3].get('greeting_response', '')
        return jsonify({'answer': answer['response']}), 200

    else:
        return jsonify({'answer': 'Cannot understand the intent. Please type a proper query'}), 200
    
@app.get('/clear_memory')
def clear_memory():
    global thread_id
    thread_id += 1
    return jsonify({"message": "Memory cleared successfully"}), 200


