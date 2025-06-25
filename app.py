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
from bs4 import BeautifulSoup
import requests
from urllib.parse import urlparse
import json
import tiktoken
from sqlalchemy import create_engine, text


app = Flask(__name__)

CORS(app, origins= "*" )

app.config['CORS_HEADERS'] = 'Content-Type'

memory = MemorySaver()
thread_id = 1

class State(TypedDict):
    intent: str
    user_query: Annotated[list, add_messages]
    pages_summary = str
    context: Annotated[list, add_messages]
    response: Annotated[list, add_messages]
    language: str

graph_builder=StateGraph(State)

def answer_query(state:State):
    if state['intent'] == "Greeting":
        print("Debug: Greeting detected, skipping SQL query.")
        return {'context': ''}  
    
    elif state['intent'] == "Query":
        query = state['user_query'][-1]

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


def translate_to_hindi(text: str) -> str:
    try:
        translation_prompt = PromptTemplate.from_template(
            """You are a professional translator. Translate the following text into Hindi.
            
            Preserve:
            - Formatting
            - Structure (headings, bullet points, etc.)
            - Punctuation
            - Names of people, places, and organizations (as appropriate)
            - Do not give the answer as here is the hindi translation. Give give response as given in the english text.
            - Preserve the gender as well.

            Text:
            {text}

            Hindi Translation:
            """
        )
        chain = translation_prompt | llm
        result = chain.invoke({'text': text})
        return result.content
    except Exception as e:
        print("Translation error:", e)
        return text 


def greeting_answer(state:State):

    language = state['language']

    qa_prompt = PromptTemplate.from_template(
      """ Your name is Bharti. You are an AI assistant for the Indian Culture Portal that deal with Indian Culture and History.
           When a greets you you should reply with a formal greeting.

           Talk about your capabilities such as search through books, Q/A through the content.
           Do not give any content here
           Add emojis wherever necessary. But not much of it.
           keep the answer short and sweet
        """
    )

    if state['intent'] == "Greeting":
        chain = qa_prompt | llm
        response = chain.invoke({'question': state['user_query']})
        content = response.content
        
        if language == 'hi':
            content = translate_to_hindi(content)

        return {'response': content}

def query_answer(state: State):
    language = state['language']
    
    qa_prompt = PromptTemplate.from_template(
        """You are an AI assistant for the Indian Culture Portal.
        Answer ONLY using the context provided. Do NOT guess or fabricate anything.
        Group your answer category-wise.
        Respond ONLY with a valid JSON array, strictly matching this structure:
        [
          {{
            "category": "Category Name",
            "description": "3-4 lines summary about this category",
            "resources": [
              {{
                "title": "Resource Title",
                "url": "category url or 'NA'"
              }}
            ]
          }}
        ]

        DO NOT include markdown, extra quotes, or any text outside the JSON array.

        context: {context}
        User question: {question}
        """
    )

    if state['intent'] == "Query":
        chain = qa_prompt | llm
        response = chain.invoke({
            'context': state['context'],
            'question': state['user_query']
        })

        try:
            parsed_json = json.loads(response.content)
            if not isinstance(parsed_json, list):
                raise ValueError("Expected a list of category dictionaries")
        except Exception as e:
            print(f"Initial Parsing Error: {e}")

            fix_json_prompt = PromptTemplate.from_template(
                """You are a JSON repair tool.
                Your task is to correct invalid JSON and return only the corrected JSON list.
                Only fix formatting issues. Just return the correct json nothing else. Do not alter the original content. 

                Input:
                {bad_json}

                Output:
                """
            )

            fix_chain = fix_json_prompt | llm
            recovery_response = fix_chain.invoke({'bad_json': response.content})

            try:
                parsed_json = json.loads(recovery_response.content)
                if not isinstance(parsed_json, list):
                    raise ValueError("Still not a list")
            except Exception as e2:
                print(f"Recovery Parsing Error: {e2}")
                parsed_json = [{
                    "category": "Invalid" if language != "Hindi" else "अमान्य",
                    "description": "Proper response not returned for the query. Try Again!" if language != "Hindi"
                                  else "प्रश्न के लिए उचित उत्तर प्राप्त नहीं हुआ। कृपया पुनः प्रयास करें!",
                    "resources": [],
                    "category_url": "NA"
                }]

        if language == "hi":
            for item in parsed_json:
                item["category"] = translate_to_hindi(item.get("category", ""))
                item["description"] = translate_to_hindi(item.get("description", ""))

        json_str = json.dumps(parsed_json, ensure_ascii=False)

        return {
            'response': [{
                'role': 'assistant',
                'content': json_str
            }]
        }

    
def truncate_text(text, max_tokens=1300):
    words = text.split()
    return ' '.join(words[:max_tokens])


def extract_page_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching page content: {e}")
        return None


def summarise_content(data, language):
    raw_text = json.dumps(data, indent=2)
    truncated_text = truncate_text(raw_text)

    print(language)

    qa_prompt = PromptTemplate.from_template(
        """You are an AI assistant for the Indian Culture Portal, specializing in Indian culture, history, and governance.

        **Instructions:**
        - Answer ONLY based on the provided context.
        - DO NOT make up information or provide random guesses.
        - If related topics are available, suggest them in a structured manner.
        - Do not provide personal opinions or views.
        - I will give you part of a page. Extract meaningful information and summarize it in 200-300 words.
        - Do not give related topics or related keywords.

        **Response Formatting:**
        - Use **bold headings** for clarity.
        - Use bullet points for listing relevant books, laws, or documents.
        - Keep the answer concise and informative.
        - Use emojis **sparingly** to enhance readability (not excessively).
        
        context: {context}
        User question: Summarize the content of this page.
        """
    )

    try:
        chain = qa_prompt | llm
        response = chain.invoke({'context': truncated_text})
        english_response = response.content

        if language == 'hi':
            return translate_to_hindi(english_response)
        
        return english_response
    except Exception as e:
        print(f"Error summarizing content: {e}")
        return "Failed to generate summary"
    

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
    language = data['language']

    config = {"configurable": {"thread_id": thread_id, "language": language}}
    events = graph.stream(
        {'user_query': [{'role': 'user', 'content': user_query}],
         'language': language},
        config = config
    )
    
    for event in events:
        events_list.append(event)

    print(events_list[0]['question_type']['intent'])

    if events_list != []:
        if events_list[0]['question_type']['intent'] == 'Query':
            answer = events_list[2].get('query_response', '') or events_list[3].get('greeting_response', '')
            answer = events_list[3].get('query_response', '') or events_list[2].get('greeting_response', '')
            print({'response': answer['response']})
            return jsonify({'answer': json.loads(answer['response'][0]['content'])}), 200

        
        elif events_list[0]['question_type']['intent'] == 'Greeting':
            answer = events_list[3].get('greeting_response', '')
            return jsonify({'answer': answer['response']}), 200

        else:
            return jsonify({'answer': 'Cannot understand the intent. Please type a proper query'}), 404
    else:
        return jsonify({'answer': 'Cannot generate response. Try Again!'}), 404


@app.post('/summarise_page')
def summarise_page_endpoint():
    request_data = request.get_json()
    url = request_data['url']

    if not url:
        return jsonify({'error': 'URL missing'}), 404
    

    # Helper functions
    def parse_url_segments(segments):
        category = ''
        page = ''
        nid = ''
        for segment in segments:
            if 'title' not in segment and 'nid' not in segment and 'page' not in segment:
                if not category:
                    category = segment.lower()
            elif 'page' in segment:
                page = int(segment.split('=')[-1]) - 1
            elif 'nid' in segment:
                nid = segment.split('=')[-1]
        return category, page, nid
    

    #------------------------------------------------------------------------------------
    def handle_musical_instruments(parsed_url, page, nid, language):
        try:
            musical_instrument = parsed_url.split('/')[2]
            print('musical_instrument', musical_instrument)
            base_url = 'https://icvtesting.nvli.in/rest-v1/musical-instruments'
            api_url = f'{base_url}/{musical_instrument}?page={page if page != "" else 0}'
            print('api_url', api_url)

            data = extract_page_content(api_url)
            if not data or 'results' not in data:
                return jsonify({"summary": "No data found"}), 404

            instrument_data = next((inst for inst in data['results'] if str(inst.get('nid')) == str(nid)), None)

            if instrument_data:
                answer = summarise_content(instrument_data, language)
                return jsonify({'summary': answer}), 200
            else:
                return jsonify({"summary": "No NID found to fetch data. Try another page"}), 404
        except Exception as e:
            print("Error in musical instrument handler:", e)
            return jsonify({'summary': 'No NID Found'}), 500


    #------------------------------------------------------------------------------------
    def handle_cultural_chronicles(category, page, nid, language):
        def try_get_page(p):
            api_url = f'https://icvtesting.nvli.in/rest-v1/{category}?page={p}'
            data = extract_page_content(api_url)
            if not data or 'results' not in data:
                return None
            return next((item for item in data['results'] if str(item.get('nid')) == str(nid)), None)

        if page == '':
            page = 0

        matched_item = try_get_page(page) or try_get_page(1 if page == 0 else 0)

        if matched_item:
            answer = summarise_content(matched_item, language)
            return jsonify({'summary': answer}), 200
        else:
            return jsonify({"summary": "No NID found to fetch data. Try another page"}), 404
        
    #------------------------------------------------------------------------------------
    def handle_textiles(parsed_url, page, nid, language):
        try:
            base_url = 'https://icvtesting.nvli.in/rest-v1/textiles-and-fabrics-of-india'
            subcategory_type = parsed_url.split('/')[2].lower()

            if subcategory_type == 'artisans':
                api_url = f'{base_url}/artisan-new??page={page if page != "" else 0}'
                print('api_url', api_url)

            elif subcategory_type == 'trade':
                api_url = f'{base_url}/trade?page={page if page != "" else 0}'
                print('api_url', api_url)

            elif subcategory_type == 'history':
                api_url = f'{base_url}/historys?page={page if page != "" else 0}&&field_state_name_value='
                print('api_url', api_url)
            
            elif subcategory_type == 'manufacturing-process':
                process_type = parsed_url.split('/')[3]
                api_url = f'{base_url}/manufacturing-technique/{process_type}?page={page if page != "" else 0}&&field_state_name_value='
                print('api_url', api_url)
            
            elif subcategory_type == 'freedom-movement-and-textiles':
                api_url = f'https://icvtesting.nvli.in/rest-v1/INDIGO-DYE-AND-REVOLT?page={page if page != "" else 0}'
                print('api_url', api_url)

            elif subcategory_type == 'artifact':
                api_url = f'{base_url}/artifacts?page=0&&field_state_name_value='
                print('api_url', api_url)


            elif subcategory_type == 'textiles-and-fabrics-of-indian-state':
                state = parsed_url.split('/')[3].split('=')[-1]
                api_url = f'{base_url}/textilestatemarker?page=0&&field_state_name_value={state}'
                print('api_url', api_url)

            data = extract_page_content(api_url)
            if not data or 'results' not in data:
                return jsonify({"summary": "No data found"}), 404

            if nid:
                subcategory_data = next((category_data for category_data in data['results'] if str(category_data.get('nid')) == str(nid)), None)
            else:
                subcategory_data = data
            
            if subcategory_data:
                answer = summarise_content(subcategory_data, language)               
                return jsonify({'summary': answer}), 200
            else:
                return jsonify({'summary': 'No NID Found to fetch data. Try another page'}), 404
        except Exception as e:
            print(e)
            return jsonify({'summary': 'Failed to summarise the page. Try again!'}), 500

    #------------------------------------------------------------------------------------
    def handle_timeless_trends(parsed_url, page, nid, language):
        try:
            base_url = 'https://icvtesting.nvli.in/rest-v1/timeless-trends'
            subcategory_type = parsed_url.split('/')[2].lower()
            if subcategory_type:
                api_url = f'{base_url}/{subcategory_type}?page={page if page != "" else 0}'
            
            if 'history-of-clothing' in subcategory_type:
                api_url = 'https://icvtesting.nvli.in/rest-v1/timeless-trends/a-brief-history-section-clothing?page=0&&field_state_name_value='

            if 'history-of-accessories' in subcategory_type:
                api_url = 'https://icvtesting.nvli.in/rest-v1/timeless-trends/a-brief-history-section-accessories?page=0&&field_state_name_value='

            if 'history-of-hairstyles' in subcategory_type:
                api_url = 'https://icvtesting.nvli.in/rest-v1/timeless-trends/a-brief-history-section-hairstyle?page=0&&field_state_name_value='

            if 'games' in subcategory_type:
                api_url = f'{base_url}/games?page={page if page != "" else 0}&&field_state_name_value='
            
            if 'snippets' in subcategory_type:
                api_url = f'{base_url}/snippets-stories?page={page if page != "" else 0}&&field_state_name_value='

            print(api_url)

            data = extract_page_content(api_url)
            if not data or 'results' not in data:
                return jsonify({"summary": "No data found"}), 404

            if nid:
                subcategory_data = next((category_data for category_data in data['results'] if str(category_data.get('nid')) == str(nid)), None)
            else:
                subcategory_data = next((category_data for category_data in data['results'] if str(category_data.get('title').lower()) == subcategory_type.replace('-', ' ').lower()), None)
            
            if subcategory_data:
                answer = summarise_content(subcategory_data, language)               
                return jsonify({'summary': answer}), 200
            else:
                return jsonify({'summary': 'No NID Found to fetch data. Try another page'}), 404


        except Exception as e:
            print(e)
            return jsonify({'summary': 'Failed to summarise the page. Try again!'}), 500


    #------------------------------------------------------------------------------------
    def handle_nizams(parsed_url, page, nid, language):
        try:
            base_url = 'https://icvtesting.nvli.in/rest-v1/jewellery-of-the-nizams'
            sub_category = parsed_url.split('/')[2].lower()
            if sub_category == 'history':
                api_url = f'{base_url}/history?page={page if page != "" else 0}'
                print('api url', api_url)
            elif sub_category == 'economy':
                api_url = f'{base_url}/economy?page={page if page != "" else 0}'
                print('api url', api_url)
            elif sub_category == 'society':
                api_url = f'{base_url}/society?page={page if page != "" else 0}'
                print('api url', api_url)
            elif sub_category == 'jewels':
                api_url = f'{base_url}/jewels?page={page if page != "" else 0}'
                print('api url', api_url)
            elif sub_category == 'anecdotes':
                api_url = f'{base_url}/anecdotes?page={page if page != "" else 0}'
                print('api url', api_url)
            elif sub_category == 'princesses':
                api_url = f'{base_url}/Princess?page={page if page != "" else 0}'
                print('api url', api_url)
            
            data = extract_page_content(api_url)
            if not data or 'results' not in data:
                return jsonify({"summary": "No data found"}), 404
            
            subcategory_data = next((category_data for category_data in data['results'] if str(category_data.get('nid')) == str(nid)), None)

            if subcategory_data:
                answer = summarise_content(subcategory_data, language)               
                return jsonify({'summary': answer}), 200
            else:
                return jsonify({'summary': 'No NID Found to fetch data. Try another page'}), 404           
        except Exception as e:
            print(e)
            return jsonify({'summary': 'Failed to summarise the page. Try again!'}), 500

#-----------------------------------------------------------------------------
    def handle_unesco(parsed_url, page, nid, language):
        try:
            base_url = 'https://icvtesting.nvli.in/rest-v1/unesco'
            sub_category = parsed_url.split('/')[2].lower()
            api_url = f'{base_url}/{sub_category}?page={page if page != "" else 0}'
            print('api url', api_url)

            data = extract_page_content(api_url)
            if not data or 'results' not in data:
                return jsonify({"summary": "No data found"}), 404
            
            print('data extracted')
            
            subcategory_data = next((category_data for category_data in data['results'] if str(category_data.get('nid')) == str(nid)), None)

            if subcategory_data:
                answer = summarise_content(subcategory_data, language)               
                return jsonify({'summary': answer}), 200
            else:
                return jsonify({'summary': 'No NID Found to fetch data. Try another page'}), 404
                
        except Exception as e:
            print(e)
            return jsonify({'summary': 'Failed to summarise the page. Try again!'}), 500
        

#-----------------------------------------------------------------------------
    def handle_forts(parsed_url, page, nid, language):
        try:
            base_url = 'https://icvtesting.nvli.in/rest-v1/forts-of-india'
            sub_category = parsed_url.split('/')[2].lower()

            if sub_category == 'discover-the-forts-of-india':
                api_url = f'{base_url}/discovering-the-forts-of-india?page={page if page != "" else 0}'
            elif sub_category == 'understanding-the-forts':
                api_url = f'{base_url}/understanding-forts?page={page if page != "" else 0}'
            elif sub_category == 'fortsandthefreedomstruggle':
                api_url = 'https://icvtesting.nvli.in/rest-v1/forts-of-india/forts-and-freedom-struggle?page=0&&field_state_name_value='
            else:
                api_url = f'{base_url}/{sub_category}?page={page if page != "" else 0}'
            
            print('api url',api_url)

            data = extract_page_content(api_url)
            if not data or 'results' not in data:
                return jsonify({"summary": "No data found"}), 404

            if nid:
                subcategory_data = next((category_data for category_data in data['results'] if str(category_data.get('nid')) == str(nid)), None)
            else:
                subcategory_data = data
            
            if subcategory_data:
                answer = summarise_content(subcategory_data, language)               
                return jsonify({'summary': answer}), 200
            else:
                return jsonify({'summary': 'No NID Found to fetch data. Try another page'}), 404

     
        except Exception as e:
            print(e)
            return jsonify({'summary': 'Failed to summarise the page. Try again!'}), 500 
#-----------------------------------------------------------------------------


    def handle_ajanta(parsed_url, data, nid, language):
        try:
            base_url = 'https://icvtesting.nvli.in/rest-v1/ajanta-landing-page'
            api_url = base_url

            if len(parsed_url.split('/')) > 2:
                sub_category = parsed_url.split('/')[2].lower()
                if sub_category == 'paintings':
                    category = 'ajanta-'+sub_category[:-1]
                    api_url = f'https://icvtesting.nvli.in/rest-v1/{category}?page={page if page != "" else 0}'
                    print(api_url)
                else:
                    category = 'ajanta-'+sub_category
                    api_url = f'https://icvtesting.nvli.in/rest-v1/{category}?page={page if page != "" else 0}'
                    print(api_url)


            data = extract_page_content(api_url)
            if not data or 'results' not in data:
                return jsonify({"summary": "No data found"}), 404

            if nid:
                subcategory_data = next((category_data for category_data in data['results'] if str(category_data.get('nid')) == str(nid)), None)
            else:
                subcategory_data = data
            
            if subcategory_data:
                answer = summarise_content(subcategory_data, language)               
                return jsonify({'summary': answer}), 200
            else:
                return jsonify({'summary': 'No NID Found to fetch data. Try another page'}), 404

        except Exception as e:
                    print(e)
                    return jsonify({'summary': 'Failed to summarise the page. Try again!'}), 500 
#-----------------------------------------------------------------------------


    def handle_virtual_walkthrough(parsed_url, page, nid, language):
        try:
            base_url = 'https://icvtesting.nvli.in/rest-v1'
            sub_category = parsed_url.split('/')[2].lower()
            if sub_category == 'virtual-walkthrough':
                api_url = f'{base_url}/{sub_category}?page={page if page != "" else 0}'
                print(api_url)

            data = extract_page_content(api_url)

            if not data or 'results' not in data:
                return jsonify({"summary": "No data found"}), 404

            subcategory_data = next((category_data for category_data in data['results'] if str(category_data.get('nid')) == str(nid)), None)

            if subcategory_data:
                answer = summarise_content(subcategory_data, language)               
                return jsonify({'summary': answer}), 200
            else:
                return jsonify({'summary': 'No NID Found to fetch data. Try another page'}), 404

        except Exception as e:
                    print(e)
                    return jsonify({'summary': 'Failed to summarise the page. Try again!'}), 500 
#-----------------------------------------------------------------------------
    
    def handle_DOD(parsed_url, page, nid, language):            
        try:
            if 'Story' in parsed_url:
                api_url = f'https://icvtesting.nvli.in/rest-v1/district-repository?page={page}&f%5B0%5D=category_ddr%3ADDR%20Story'
            elif 'Traditions' in parsed_url:
                api_url = f'https://icvtesting.nvli.in/rest-v1/district-repository?page={page}&f%5B0%5D=category_ddr%3ATraditions%20%26%20Art%20Forms'
            elif 'Personality' in parsed_url:
                api_url = f'https://icvtesting.nvli.in/rest-v1/district-repository?page={page}&f%5B0%5D=category_ddr%3APersonality'
            elif 'Events' in parsed_url:
                api_url = f'https://icvtesting.nvli.in/rest-v1/district-repository?page={page}&f%5B0%5D=category_ddr%3AEvents'
            elif 'Treasures' in parsed_url:
                api_url = f'https://icvtesting.nvli.in/rest-v1/district-repository?page={page}&f%5B0%5D=category_ddr%3AHidden%20Treasures'

            print(api_url)

            data = extract_page_content(api_url)

            if not data or 'results' not in data:
                return jsonify({"summary": "No data found"}), 404

            subcategory_data = next((category_data for category_data in data['results'] if str(category_data.get('nid')) == str(nid)), None)

            if subcategory_data:
                answer = summarise_content(subcategory_data, language)               
                return jsonify({'summary': answer}), 200
            else:
                return jsonify({'summary': 'No NID Found to fetch data. Try another page'}), 404
          
        except Exception as e:
            print(e)
            return jsonify({'summary':'Failed to summarise the page. Try again!'}), 500 
#-----------------------------------------------------------------------------

    def handle_artifacts(parsed_url, page, nid, language):
        try:
            base_url = 'https://icvtesting.nvli.in/rest-v1/retrieved-artefacts-of-india'
            sub_category = parsed_url.split('/')[2].lower()
            
            if sub_category in ['reclaimed-relics', 'artefact-chronicles']:
                api_url = f'{base_url}/{sub_category}?page={page if page != "" else 0}&&field_state_name_value='
                print(api_url)

                data = extract_page_content(api_url)
                if not data or 'results' not in data:
                    return jsonify({"summary": "No data found"}), 404

                subcategory_data = next((category_data for category_data in data['results'] if str(category_data.get('nid')) == str(nid)), None)

                if subcategory_data:
                    answer = summarise_content(subcategory_data, language)               
                    return jsonify({'summary': answer}), 200
                else:
                    return jsonify({'summary': 'No NID found to fetch data. Try another page'}), 404
            else:
                return jsonify({'summary: ' 'The page does not contain information to summarise'}), 404
            
        except Exception as e:
            print(e)
            return jsonify({'summary': 'Failed to summarise the page. Try again!'}), 500 
#-----------------------------------------------------------------------------

    def handle_freedom_fighters(parsed_url, page, nid, language):
        try:
            base_url = 'https://icvtesting.nvli.in/rest-v1/freedom-archive' 
            sub_category = parsed_url.split('/')[2].lower()

            if sub_category == 'unsung-heroes':
                api_url = f'https://icvtesting.nvli.in/rest-v1/unsung-heroes?page={page if page != "" else 0}&&field_state_name_value='
            if sub_category == 'historic-cities':
                city = parsed_url.split('/')[3].lower()
                api_url = f'https://icvtesting.nvli.in/rest-v1/{city}/Historic_Cities_Freedom_Movement?page=0&&field_state_name_value='
            if sub_category == 'forts':
                api_url = f'https://icvtesting.nvli.in/rest-v1/forts-of-india/forts-and-freedom-struggle?page=0&&field_state_name_value='
            if sub_category == 'textile':
                api_url = f'https://icvtesting.nvli.in/rest-v1/INDIGO-DYE-AND-REVOLT?page=0&&field_state_name_value='
            if sub_category == 'historic-cities':
                city = parsed_url.split('/')[3].lower()
                api_url = f'https://icvtesting.nvli.in/rest-v1/{city}/Historic_Cities_Freedom_Movement?page=0&&field_state_name_value='
    
            print(api_url)

            data = extract_page_content(api_url)

            if not data or 'results' not in data:
                return jsonify({"summary": "No data found"}), 404

            if nid:
                subcategory_data = next((category_data for category_data in data['results'] if str(category_data.get('nid')) == str(nid)), None)
            else:
                subcategory_data = data
            
            if subcategory_data:
                answer = summarise_content(subcategory_data, language)               
                return jsonify({'summary': answer}), 200
            else:
                return jsonify({'summary': 'No NID Found to fetch data. Try another page'}), 404

        
        except Exception as e:
            print(e)
            return jsonify({'summary': 'Failed to summarise the page. Try again!'}), 500 
#-----------------------------------------------------------------------------

    # def handle_global_search(parsed_url, page, nid, language):


# Function calling
#-----------------------------------------------------------------------------
    try:
        parsed_url = urlparse(url).path
        print('debug',parsed_url)

        lang = 'hi' if parsed_url.split('/')[1] == 'lang=hi' else 'en'
        print(lang)

        if lang == 'hi':
           parsed_url = parsed_url.replace('/lang=hi', '')

        segments = [s for s in parsed_url.split('/') if s]
        category, page, nid = parse_url_segments(segments)
    

        if category == 'musical-instruments-of-india':
            return handle_musical_instruments(parsed_url, page, nid, language=lang)
        elif category == 'textiles-and-fabrics-of-india':
            return handle_textiles(parsed_url, page, nid, language=lang)
        elif category == 'timeless-trends':
            return handle_timeless_trends(parsed_url, page, nid, language=lang)
        elif category == 'jewellery-of-the-nizams':
            return handle_nizams(parsed_url, page, nid, language=lang)
        elif category == 'unesco':
            return handle_unesco(parsed_url, page, nid, language=lang)
        elif category == 'forts-of-india':
            return handle_forts(parsed_url, page, nid, language=lang)
        elif category == 'ajanta-caves':
            return handle_ajanta(parsed_url, page, nid, language=lang)
        elif category == '3d-explorations':
            return handle_virtual_walkthrough(parsed_url, page, nid, language=lang)
        elif category == 'districts-of-defiance':
            return handle_DOD(parsed_url, page, nid, language=lang)
        elif category == 'retrieved-artefacts-of-india':
            return handle_artifacts(parsed_url, page, nid, language=lang)
        elif category == 'freedom-archive':
            return handle_freedom_fighters(parsed_url, page, nid, language=lang)
        else:
            return handle_cultural_chronicles(category, page, nid, language=lang)
    except Exception as e:
            print(e)


#------------------------------------------------------------------------------------
@app.get('/clear_memory')
def clear_memory():
    global thread_id
    thread_id += 1
    return jsonify({"message": "Memory cleared successfully"}), 200
