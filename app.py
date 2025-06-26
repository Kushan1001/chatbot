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

    def handle_food_and_culture(parsed_url, page, nid, language):
        try:
            base_url = 'https://icvtesting.nvli.in/rest-v1/food-and-culture'
            sub_category = parsed_url.split('/')[2].lower()
            if sub_category == 'cuisines-of-india':
            
                subcategory_data = [
                        {
                            "title": "The Food of Maharashtra: A Sweet and Tangy Journey",
                            "snippet": "The Food of Maharashtra: A Sweet and Tangy Journey\n\n\n\n\n\n\n\n\n\n\nA Maharashtrian woman, by Raja Ravi Varma"
                        },
                        {
                            "title": "Rajasthani Cuisine: A Fusion of Resilience, a Royal Past and Innovation",
                            "snippet": "Rajasthani Cuisine: A Fusion of Resilience, a Royal Past and Innovation\n\n\n\n\n\n\nRajasthan, the land of royals, is one of the most popular tourist destinations of India. Every year, visitors from all over the world throng into this beautiful state on the north-western frontier of India to marvel at its fascinating landscape, colourful art and crafts, exotic songs and dances, and exquisite historical monuments. The food of this land is also equally amazing and delightful. Born out of the exigencies of arid land, a harsh climate and a war-torn past, the cuisine of Rajasthan truly captures the spirit of resilience as well as imagination of a people in the face of all odds."
                        },
                        {
                            "title": "Goan Cuisine: A Confluence of Cultures ",
                            "snippet": "Goan Cuisine: A Confluence of Cultures\n\n\n\n\n\n\nThe unique cuisine of Goa developed out of a merger of various cultures that it came into contact with over the centuries such as the Portuguese, Arab, Brazilian, African, French, Konkani, Malabari, Malaysian and Chinese. The three major communities of Goa - Hindus, Muslims, and Christians, contribute to the culinary tradition in their own manner. The Konkan farmers and fishermen consume fish and rice on a wide scale. The Christian community patronises items such as beef, seafood, and pork. The intermixing of multiple cultural elements is mirrored within the cuisine of Goa in a distinctive mix of richness and subtlety."
                        },
                        {
                            "title": "Kerala Cuisine: A Melting Pot ",
                            "snippet": "Kerala Cuisine: A Melting Pot\n\n\n\n\n\n\nKerala, in the south-western part of India, is known for its rich heritage and cultural diversity. Situated along the Malabar coast, Kerala has had regular interaction with the West since ancient times. From the coming of the Arab traders to the Portuguese, and later the British, Kerala has witnessed it all. This greatly influenced the socio-cultural fabric of the region, making it one of the most diverse states of India."
                        },
                        {
                            "title": "Andhra Cuisine: A Symphony of Spices",
                            "snippet": "Andhra Cuisine: A Symphony of Spices\n\n\n\n\n\n\n\n\nAn Andhra platter. Image source: Wikimedia Commons"
                        },
                        {
                            "title": "The Cuisine of Tamil Nadu: Beyond Sambar and Filter Coffee",
                            "snippet": "The Cuisine of Tamil Nadu: Beyond Sambar and Filter Coffee\n\n\n\n\nTamil Nadu, the southern-most state of India, is known for its rich cultural heritage and magnificent temples that stand tall in its various cities and towns. Culture is deeply rooted among the Tamilians with most of them involved in one art form or the other like Carnatic music or classical dance, or even preparing traditional food items in the strictly prescribed manner. The cuisine of Tamil Nadu is a reflection of the various influences that the state has come to assimilate over the centuries. From the early Cholas to the Marathas of Tanjore, each dynasty left a mark on this exquisite cuisine. With an equal number of vegetarian and non-vegetarian dishes, this cuisine is famous for its simplicity, rich flavours, and generous use of spices.\n\n\n\n\n\n\nGeography and staples"
                        },
                        {
                            "title": "Lakshadweep Cuisine: The Sea on a Plate ",
                            "snippet": "Lakshadweep Cuisine: The Sea on a Plate\n\n\n\n\n\n\nLakshadweep, a group of 36 islands, is located off the coast of Kerala. The term “lakshadweep” literally means “thousand islands” in Malayalam and Sanskrit. Out of the many small islands, only 10 continue to be inhabited and only a few are allowed to be visited by tourists. The inhabited islands are Agatti, Kalpeni, Kadmat, Kiltan, Cheltat, Amini, Bitra, Androth, Minicoy and Kavaratti, (the capital). Known for its clear blue waters and simple lifestyle, this Union Territory of India never fails to attract tourists from various parts of the globe."
                        },
                        {
                            "title": "Odisha",
                            "snippet": "Odisha\n\n\n\n\n\n\n \n\n\n\n\nOdisha Cuisine"
                        },
                        {
                            "title": "The Land of Bihar and its Wholesome Food ",
                            "snippet": "The Land of Bihar and its Wholesome Food\n\n\n\n\n\n\nThe state of Bihar is situated in the eastern region of the Indian mainland. This landlocked region is famous for its ancient traditions and heritage sites including Bodh Gaya, where Buddha attained enlightenment, the ancient Nalanda University, for the sweet and lilting Bhojpuri language, and much more. While Bihari cuisine has many distinctive dishes, unfortunately, they are not widely known in the rest of the country."
                        },
                        {
                            "title": "Manipuri Cuisine: A Unique Experience in Earthy Flavours",
                            "snippet": "Manipuri Cuisine: A Unique Experience in Earthy Flavours\n\n\n\n\n\n\nThe cuisine of Manipur reflects the geographical and socio-cultural peculiarities of this land situated in the North-Eastern part of the Indian subcontinent. The culinary fare of this region reflects the intimate connection of its people with nature. With an exciting ensemble of flavours ranging from plain to piquant, Manipuri food is an absolute delight to the senses."
                        },
                        {
                            "title": "The Culinary Treasures of Sikkim",
                            "snippet": "The Culinary Treasures of Sikkim\n\n\n\n\n\n\nThe erstwhile Himalayan kingdom of Sikkim became part of the Indian Union in 1975. As part of the Eastern Himalayas, the hilly terrain of Sikkim rises from the tropical jungles at the foothills and ascends to high alpine valleys and lofty peaks. Kanchenjunga, the third highest peak in the world is located at the Singalila range which forms a boundary between Nepal and Sikkim. This mountain state is bounded by Tibet on the north, Bhutan on the east and south and Nepal on the west. This particular location of Sikkim has given the state a multicultural and multi-ethnic character. The main ethnic groups in the Sikkim hills are the Lepchas, Bhutias and Nepalis and accordingly the cuisine of Sikkim is representative of these communities. Over time communities from the mainland have migrated to Sikkim and have brought with them their special foods to the hills. The food culture in the Eastern Himalayas of Sikkim has evolved over a period of time based on environmental, social and cultural factors and certain dishes have transcended cultural and territorial borders and have come to be embraced by all communities in Sikkim. These include dishes like the \npatlesishnu\n or \nsouchya\n (nettle soup), \nthukpa\n (wheat noodle soup) and \ntiteningro\n (fiddlehead fern curry with \nchurpi\n or cottage cheese)."
                        },
                        {
                            "title": "Arunachalee Cuisine: Frontier Fare",
                            "snippet": "Arunachalee Cuisine: Frontier Fare\n\n\n\n\n\n\nThe Indian state of Arunachal Pradesh, formerly a part of the North East Frontier Agency or NEFA, is crisscrossed by five big rivers, the T\nirap, Lohit, Siang, Subansiri,\n and \nKameng\n. This large territory has rich flora and fauna thanks to the climatic conditions that range from temperate to tropical to alpine. The luxuriant growth of its forest cover can be attributed to the heavy rainfall it receives and the associated high levels of humidity. As this most expansive North-eastern state is endowed with rich bio-resources and abundant forest cover, the majority of its tribal population lives in or in close proximity to the forests and depends on the forest produces for its sustenance. This ‘Land of the Rising Sun’ in the Eastern Himalayas is bountiful not only in natural resources, but is also home to around twenty-six major tribal communities, like the \nMonpa, Sherdukpen, Aka, Khowa, Apatani, Khampti,\n and \nGalo\n that have their own distinct languages, dances, music, oral traditions, arts, crafts, and cuisine."
                        },
                        {
                            "title": "Meghalaya Cuisine",
                            "snippet": "Meghalaya Cuisine\n\n\n\n\n\n\n\n\n\n\nSohra (Cherrapunji), Meghalaya. Image Source: Biri Jumsi"
                        },
                        {
                            "title": "Flavours from the Axomiya Akholghor",
                            "snippet": "Flavours from the Axomiya Akholghor\n\n\n\n\n\n\nThe Axomiya Akholghor (traditional Assamese kitchen) has churned out a myriad of earthy delicacies that reflect the ingenuity of the people of this ecologically rich land. Yet, Assamese cuisine is still a largely uncharted territory for people outside North-East India. The Assamese are often credited with having an “adventurous palate”, a notion which deters many (cautious) food lovers from trying out this wholesome cuisine. However, Assamese cuisine is not as removed from culinary traditions in other parts of India, as it is sometimes made out to be. At the same time, for a connoisseur of food, this cuisine has a lot of novelty to offer. The food of this land is a precious blend of the ordinary and the unique."
                        },
                        {
                            "title": "Naga Cuisine (A Feast for the Senses)",
                            "snippet": "Naga Cuisine\n\n(A Feast for the Senses)"
                        },
                        {
                            "title": "Thevo Chu, Pork meat with Axone",
                            "snippet": ""
                        },
                        {
                            "title": "The Mizo Food Ethic: Simplicity and Selflessness",
                            "snippet": "The Mizo Food Ethic: Simplicity and Selflessness\n\n\n\n\n\n\n\n\n\n\nA traditional Mizo spread"
                        },
                        {
                            "title": "Tripura Cuisine",
                            "snippet": "Tripura Cuisine"
                        }
                    ]

            if subcategory_data:
                answer = summarise_content(subcategory_data, language)               
                return jsonify({'summary': answer}), 200
            else:
                return jsonify({'summary': 'No NID Found to fetch data. Try another page'}), 404

        except Exception as e:
                    print(e)
                    return jsonify({'summary': 'Failed to summarise the page. Try again!'}), 500
#-----------------------------------------------------------------------------
            

    def handle_festivals(parsed_url, page, nid, language):
        try:
            base_url = 'https://icvtesting.nvli.in/rest-v1/festivals-of-india' 
            sub_category = parsed_url.split('/')[2].lower()

            if sub_category == 'fairs-and-pilgrimages':
                api_url = f'https://icvtesting.nvli.in/rest-v1/festivals-of-india/fairs-and-pilgrimages?page={page if page != "" else 0}&&field_state_name_value='
            if sub_category == 'folk-festivals':
                city = parsed_url.split('/')[3].lower()
                if city == 'honoring-deities':
                    api_url = 'https://icvtesting.nvli.in/rest-v1/festivals-of-India/folk-festivals/Honouring-Deities?page=0&&field_state_name_value='
                if city == 'social-traditions':
                    api_url = 'https://icvtesting.nvli.in/rest-v1/festivals-of-India/folk-festivals/Social-Traditions?page=0&&field_state_name_value='
                if city == 'celebrating-nature':
                    api_url = 'https://icvtesting.nvli.in/rest-v1/festivals-of-India/folk-festivals/Celebrating-Nature?page=0&&field_state_name_value='
            if sub_category == 'tribal-festivals':
                api_url = f'https://icvtesting.nvli.in/rest-v1/festivals-of-India/tribal-festivals/worshipping-nature?page=0&&field_state_name_value='
           
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
    # States of India
    def handle_states(parsed_url, page, nid, language):
        try:
            base_url = 'https://icvtesting.nvli.in/rest-v1/festivals-of-india' 
            category = parsed_url.split('/')[2].lower().strip()
            sub_category = parsed_url.split('/')[3].lower().strip()
            section = parsed_url.split('/')[4].lower().strip()

            if category == 'north-east-archive':
                if sub_category == 'unsung-heroes':
                    api_url = f'https://icvtesting.nvli.in/rest-v1/north-east-archive/unsung-heroes?page=0&&field_state_name_value=?page={page if page != "" else 0}&&field_state_name_value='
                elif sub_category == 'tales-from-the-hinterland':
                    api_url = f'https://icvtesting.nvli.in/rest-v1/tales-from-the-hinterland?page={page if page != "" else 0}&&field_state_name_value='

            if category == 'east-india':
                if sub_category == 'bihar':
                    if section == 'tidbits-tales-and-trivis':
                        api_url = f'https://icvtesting.nvli.in/rest-v1/states-of-india/bihar/tidbits-tales-trivia?page=0&&field_state_name_value='
                    elif section == 'bihar-through-traveller-s-gaze':
                        api_url = 'https://icvtesting.nvli.in/rest-v1/states-of-india/bihar/bihar-through?page=0&&field_state_name_value='
                    elif section == 'art-and-architecture':
                        api_url = 'https://icvtesting.nvli.in/rest-v1/states-of-india/bihar/art-architecture?page=0&&field_state_name_value='
                    elif section == 'freedom-archive':
                        api_url  = 'https://icvtesting.nvli.in/rest-v1/states-of-india/bihar/freedom-archive?page=0&&field_state_name_value='

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

    # Iconic battles
    def handle_iconic_battles(parsed_url, page, nid, language):
        try:
            api_url = f'https://icvtesting.nvli.in/rest-v1/iconic-battle-of-india/detail?page={page if page != "" else 0}&&field_state_name_value='

            data = extract_page_content(api_url)
            if not data or 'results' not in data:
                return jsonify({"summary": "No data found"}), 404

            subcategory_data = next((category_data for category_data in data['results'] if str(category_data.get('nid')) == str(nid)), None)

            if subcategory_data:
                answer = summarise_content(subcategory_data, language)               
                return jsonify({'summary': answer}), 200
            else:
                return jsonify({'summary': 'No NID found to fetch data. Try another page'}), 404
        except Exception as e:
            print(e)
            return jsonify({'summary': 'Failed to summarise the page. Try again!'}), 500 
#-----------------------------------------------------------------------------

    # lengendary figures
    def handle_legendary_figures(parsed_url, page, nid, language):
        try:
            api_url = f'https://icvtesting.nvli.in/rest-v1/legendary-figure/kings-queens?page={page if page != "" else 0}&&field_state_name_value='

            data = extract_page_content(api_url)
            if not data or 'results' not in data:
                return jsonify({"summary": "No data found"}), 404

            subcategory_data = next((category_data for category_data in data['results'] if str(category_data.get('nid')) == str(nid)), None)

            if subcategory_data:
                answer = summarise_content(subcategory_data, language)               
                return jsonify({'summary': answer}), 200
            else:
                return jsonify({'summary': 'No NID found to fetch data. Try another page'}), 404
        except Exception as e:
            print(e)
            return jsonify({'summary': 'Failed to summarise the page. Try again!'}), 500 
#-----------------------------------------------------------------------------

    # folk tales of India
    def handle_folktalesofindia(parsed_url, page, nid, language):
        try:
            api_url = f'https://icvtesting.nvli.in/rest-v1/fairytales-landing-main?page={page if page != "" else 0}&&field_state_name_value='

            data = extract_page_content(api_url)
            if not data or 'results' not in data:
                return jsonify({"summary": "No data found"}), 404

            subcategory_data = next((category_data for category_data in data['results'] if str(category_data.get('nid')) == str(nid)), None)

            if subcategory_data:
                answer = summarise_content(subcategory_data, language)               
                return jsonify({'summary': answer}), 200
            else:
                return jsonify({'summary': 'No NID found to fetch data. Try another page'}), 404
        except Exception as e:
            print(e)
            return jsonify({'summary': 'Failed to summarise the page. Try again!'}), 500 
#-----------------------------------------------------------------------------
    
    # healing through ages
    def handle_healing_through_the_ages(parsed_url, page, nid, language):
        try:
            api_url = f'https://icvtesting.nvli.in/rest-v1/healing-through-the-ages/pan-india-traditions?page={page if page != "" else 0}&&field_state_name_value='

            data = extract_page_content(api_url)
            if not data or 'results' not in data:
                return jsonify({"summary": "No data found"}), 404

            subcategory_data = next((category_data for category_data in data['results'] if str(category_data.get('nid')) == str(nid)), None)

            if subcategory_data:
                answer = summarise_content(subcategory_data, language)               
                return jsonify({'summary': answer}), 200
            else:
                return jsonify({'summary': 'No NID found to fetch data. Try another page'}), 404
        except Exception as e:
            print(e)
            return jsonify({'summary': 'Failed to summarise the page. Try again!'}), 500
#-----------------------------------------------------------------------------
    # classical dances
    def handle_classical_dances(parsed_url, page, nid, language):
        try:
            api_url = f'https://icvtesting.nvli.in/rest-v1/classical-dances-details?page={page if page != "" else 0}&&field_state_name_value='
            print('api_url',api_url)
            data = extract_page_content(api_url)
            if not data or 'results' not in data:
                return jsonify({"summary": "No data found"}), 404

            subcategory_data = next((category_data for category_data in data['results'] if str(category_data.get('nid')) == str(nid)), None)

            if subcategory_data:
                answer = summarise_content(subcategory_data, language)               
                return jsonify({'summary': answer}), 200
            else:
                return jsonify({'summary': 'No NID found to fetch data. Try another page'}), 404
        except Exception as e:
            print(e)
            return jsonify({'summary': 'Failed to summarise the page. Try again!'}), 500


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
        elif category == 'food-and-culture':
            return handle_food_and_culture(parsed_url, page, nid, language=lang)
        elif category == 'festivals-of-india':
            return handle_festivals(parsed_url, page, nid, language=lang)
        elif category == 'states':
            return handle_states(parsed_url, page, nid, language=lang)
        elif category == 'iconic-battles-of-india':
            return handle_iconic_battles(parsed_url, page, nid, language=lang)
        elif category == 'handle_legendary_figures':
            return handle_legendary_figures(parsed_url, page, nid, language=lang)
        elif category == 'folktalesofindia':
            return handle_folktalesofindia(parsed_url, page, nid, language=lang)
        elif category == 'healing-through-the-ages':
            return handle_healing_through_the_ages(parsed_url, page, nid, language=lang)
        elif category == 'classical-dances-of-india':
            return handle_classical_dances(parsed_url, page, nid, language=lang)
        else:
            return handle_cultural_chronicles(parsed_url, page, nid, language=lang)
    except Exception as e:
            print(e)

            
#------------------------------------------------------------------------------------
@app.get('/clear_memory')
def clear_memory():
    global thread_id
    thread_id += 1
    return jsonify({"message": "Memory cleared successfully"}), 200
