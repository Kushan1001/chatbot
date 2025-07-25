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
import html

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

def clean_html(html_text):
    decoded_html = html.unescape(html_text)
    soup = BeautifulSoup(decoded_html, 'html.parser')
    for tag in soup(['style', 'script']):
        tag.decompose()
    text = soup.get_text(separator='\n')
    return re.sub(r'\n\s*\n', '\n\n', text).strip()


def remove_src_attributes(html_text):
    cleaned_html = re.sub(r'\s*src="[^"]*"', '', html_text)
    return cleaned_html


def clean_html_truncate(html_text, max_words=250):
    soup = BeautifulSoup(html_text or "", "html.parser")
    plain_text = soup.get_text(separator=" ")
    words = plain_text.split()
    if len(words) > max_words:
        return " ".join(words[:max_words])
    return plain_text


def clean_and_truncate_html(html_text, word_limit=100):
    clean_text = re.sub('<.*?>', '', html_text)
    clean_text = clean_text.replace("&nbsp;", " ").strip()
    clean_text = re.sub(r'\s+', ' ', clean_text)
    words = clean_text.split()
    truncated_words = words[:word_limit]
    return ' '.join(truncated_words)

def truncate_context(text_or_list, max_words: int = 700) -> str:
    if isinstance(text_or_list, list):
        text = " ".join(str(item) for item in text_or_list)
    else:
        text = str(text_or_list)
    words = text.split()
    return ' '.join(words[:max_words])


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
    

def truncate_text(text, max_tokens=850):
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

#------------------------------------------------------------------------------------------------
def summarise_content(data, language):
    raw_text = json.dumps(data, indent=2)
    truncated_text = truncate_text(raw_text)

    print('Language Detected:',  language)

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
    

# Intent Classification
#------------------------------------------------------------------------------------------------
def identify_intent(state: State):
    latest_question = (
        state["user_query"][-1].content
        if state["user_query"]
        else ""
    )

    # Build the prompt template
    classification_prompt = PromptTemplate.from_template(
        """
            You are an intent classification model for the Indian Culture Portal (IPC).
            Classify the user query into one of these intents:

            1. Greeting:
            - The user greets you (e.g., "Hello", "Hi").
            2. General:
            - The user asks about your capabilities, who you are, or general questions not requiring data lookup.
            - Includes administrative questions about the portal itself and categories description.(e.g., "Who developed this portal?", "What can you do?", "What is NVLI").
            3. Specialised:
            - The user asks for specific information about Indian culture, history, books, or content that requires searching databases or knowledge content.
            - Example: "Tell me about Mughal architecture."
            4. Unknown:
            - The query is unclear or does not fit any category.

            **Examples:**

            - "Hi there" -> Greeting
            - "What can you do?" -> General
            - "Explain Vedic literature." -> Specialised
            - "asdlkj" -> Unknown

            User Query:
            {latest_question}

            Respond with only the intent name (Greeting, General, Specialised, Unknown). No explanation.
            """
    )

    chain = classification_prompt | llm

    response = chain.invoke({"latest_question": latest_question})
    intent = response.content.strip()

    print("Intent Classification Response:", intent)
    return {"intent": intent}


# handles greeting response
#------------------------------------------------------------------------------------------------
def greeting_answer(state:State):

    language = state['language']

    qa_prompt = PromptTemplate.from_template(
      """ Your name is Bharti. You are an AI assistant for the Indian Culture Portal that deal with Indian Culture and History.
           When a greets you you should reply with a formal greeting.

           Talk about your capabilities:  search through books, Q/A through the content, summarise the information.
           Do not give any content here
           Add emojis wherever necessary. But not much of it.
           keep the answer short and sweet
           List capabilities in points but not give description of it.
        """
    )

    if state['intent'] == "Greeting":
        chain = qa_prompt | llm
        response = chain.invoke({'question': state['user_query']})
        content = response.content
        
        if language == 'hi':
            content = translate_to_hindi(content)

        return {'response': content}
    

# handles general query response
#------------------------------------------------------------------------------------------------
def general_query_answer(state: State):
    language = state['language']

    knowledge_context = """
        Recognizing the ongoing need to position itself for the digital future, 
        Indian Culture is an initiative by the Ministry of Culture. A platform that 
        hosts data of cultural relevance from various repositories and institutions all 
        over India.
        The Indian Culture Portal is a part of the National Virtual Library of India project, 
        funded by the Ministry of Culture, Government of India. The portal has been created and 
        developed by the Indian Institute of Technology (IIT), Bombay. 
        The content is available both in English and Hindi

        
        Categories available on the website:
            1. Original Catgegories (Prepared with original research by reserachers done here):
                subcategory 1: Cultural Expressions
                List of them:
                    Classical Dances of Indian (India's classical dances are a vibrant expression of its 
                    diverse cultural heritage, deeply rooted in tradition, mythology, and spiritual practice. 
                    Each form tells stories through graceful movements, intricate rhythms, and expressive gestures. 
                    These dance forms not only reflect regional identities but also embody centuries-old philosophies 
                    and artistic disciplines. This category explores the origins, evolution, and unique characteristics of 
                    India's classical dance styles, offering a window into the artistic soul of the nation.),
                    
                    Festivals Of India (To experience the festivals of India is to experience the grandeur 
                    and richness of the Indian cultural heritage. The festivals of India thrive in a culture 
                    of diversity, and the celebration of these festivals has become a time for cross-cultural 
                    exchanges. Filled with rituals, music, performances, culinary treats, and more, each 
                    festival presents its own fascinating history and unique charm. A large diversity of 
                    customs, traditions, and tales are also associated with festivals. Learn about the 
                    cultural diversity, customs and traditions, as well as the fascinating stories 
                    associated with the festivals presented in the categories below, or explore the vibrant 
                    festivals of the states by clicking on the map or finding your favourite festival.), 
                    
                    Food And Culture(The Indian culinary repertoire reflects the cultural diversity of the 
                    country. The term “Indian food” denotes a mélange of flavours from different parts of 
                    the country and showcases centuries of cultural exchange with the far corners of the 
                    world. Here, on our portal, we are making a small effort of gradually building a 
                    treasure trove of information about the countless exquisite flavours of our country. 
                    It is an ongoing venture and over time we aim to capture as much as possible of the 
                    incredible culinary diversity of this land.), Musical Instruments Of India

                    Musical Instruments of India(The Musical Instruments section of the Indian Culture 
                    portal contains information about a range of instruments from across India. 
                    The Indian Culture portal has researched and is happy to present information 
                    about the countless exquisite musical instruments of our country.)

                    Textiles and fabrics of India (Textiles and Fabrics of India is an attempt to showcase 
                    and celebrate the long and diverse tradition of Textiles in India. The history of this 
                    craft goes back to the ancient period. This section highlights and honours the 
                    craftsmanship of the Indian handloom workers, embroiderers, block printers, painters 
                    and others who have immensely contributed to build a distinct textile industry for India.)

                    Timeless Trends (n both its traditional and modern manifestations, Indian art exhibits a 
                    powerful sense of design and a vivid imagination. These are reflected in sculptures, 
                    paintings, murals, architecture, coins, and items of personal adornment like jewellery, 
                    clothing, and more. Surviving the vagaries of time, many of these artefacts are now 
                    preserved in museums, archaeological sites and cultural institutions. These seemingly 
                    ordinary artefacts act as a repository of knowledge that conveys information about the 
                    society of their time. Timeless Trends celebrates the interconnectedness of the past and 
                    the present and attempts to discover the links between the cultures and traditions we 
                    cherish, the structures and sites that dot our modern landscapes, and the little things 
                    that we do and say every day.) 
                    
                subcategory 2: Legends and Legacies
                    Folktales Of India (India has a rich and diverse tradition of folktales, shaped by its many languages, 
                    cultures, and regions. These stories—ranging from fables and fairytales to myths and legends—have been 
                    passed down through generations, reflecting the values, beliefs, and imaginations of the people. 
                    Fables use animal characters to teach moral lessons, while fairytales often involve magical beings, 
                    heroic quests, and transformations. Myths recount the exploits of gods and divine beings, explaining 
                    creation, duty, and cosmic order, whereas legends celebrate the lives of saints, warriors, poets, and 
                    jesters whose deeds live on in collective memory. Whether shared in gatherings, temples, courts, or 
                    classrooms, these tales continue to captivate audiences, preserving the spirit of India's vibrant 
                    storytelling heritage.)
                    
                    Healing Through The Ages(The 'Healing Through the Ages' category aims to trace the various dimensions and understanding of ailments and cures across India. It is a repository 
                    which brings together the different meanings of 'health' and provides an overview of both conventional 
                    and unconventional approaches to maintaining balance and restoring a sense of well-being. It will help 
                    you traverse historical, regional and cultural boundaries, and help you to cultivate a nuanced 
                    understanding of suffering and healing.)

                    Iconic Battles Of India(Warfare has shaped the course of Indian history. The subcontinent 
                    has witnessed epic battles that not only altered its destiny but also influenced the world 
                    at large. This section delves into twelve iconic battles that changed the tide of Indian 
                    history, tracing the evolution of warfare across different eras. Each of these conflicts 
                    marked the rise or fall of dynasties, introduced new systems of governance, and gave birth 
                    to lasting traditions, beliefs, and cultural patterns that continue to shape India's 
                    identity today.)
                     
                    Jewellery Of The Nizams
                    
                    Legendary Figures Of India(Throughout its rich and diverse history India has been home 
                    to towering personalities whose contributions transcended their time. This category 
                    explores the lives and legacies of such extraordinary individuals who have profoundly 
                    shaped India's history through their vision courage and intellect. Spanning a diverse 
                    spectrum of emperors spiritual leaders social reformers scholars and freedom fighters 
                    these iconic personalities represent the enduring spirit of resilience innovation and 
                    leadership. Their contributions not only influenced the course of the nation's political 
                    and cultural development but also continue to inspire generations with their unwavering 
                    commitment to justice knowledge and progress. Through their remarkable journeys they have 
                    helped define the soul of India and left an indelible imprint on its collective memory.)
                
                subcategory 3: Pan India Explorations
                    Historic Cities Of India(The map of India is dotted with cities that so many of us call home. Many of these cities have origins in our collective history. While they may now be modern and dynamic centres, they continue to represent centuries of culture and heritage that even today, sets them apart from every other city across the globe. Explore these unique urban centres and everything that they have to offer at your own pace, through a virtual expedition. Click on the icons to the right to begin a virtual visit to these historic cities! Each city has its own story, one that is told here through a collection of rare photographs, multimedia, specially-narrated tales, and more. We invite you to sift through them, explore, and discover your own favourite stories about every city.)
                    States Of India(India, a vast and vibrant nation, is a mosaic of diverse states, each woven with a unique thread of culture, history, and tradition. From the majestic, snow-capped mountains in the north to the sun-drenched coastlines in the south, each state offers a rich and varied blend of languages, cuisines, arts, and festivals. This category delves into the architectural marvels that adorn the country, the profound literary contributions from various corners, and the abundant interesting anecdotes that shape each state’s identity. It also meticulously charts the historical development of these regions through different ages, reflecting India's millennia- old heritage. Each state, with its distinct character, forms a thread in the extraordinary fabric of India.)
                    Unesco 
                
                subcategory 4: Built Heritage
                    3d Explorations(The Indian culinary repertoire reflects the cultural diversity of the country. The term “Indian food” denotes a mélange of flavours from different parts of the country and showcases centuries of cultural exchange with the far corners of the world. Here, on our portal, we are making a small effort of gradually building a treasure trove of information about the countless exquisite flavours of our country. It is an ongoing venture and over time we aim to capture as much as possible of the incredible culinary diversity of this land.)
                    Ajanta Caves(The Ajantā caves are rock-cut Buddhist cave temples carved out of a horseshoe shaped valley near the Waghora river at the edge of the Indyadhri range. The caves are a UNESCO World Heritage site and are thronged by thousands of tourists who come to admire its serene location, rock-cut architecture and beautiful Buddhist paintings that are found in the caves. These 30 rock-cut caves are part of a constellation of Buddhist cave temples dotting the Sahayādri or Western Ghats in Maharashtra. But Ajantā is unique as it hosts the finest specimens of art - Cave 9 and 10 contain the oldest Buddhist narrative paintings in India.)
                    Forts Of India(The Forts of India are some of the most awe-inspiring monuments found in the country. From the Himalayas to the peninsular tip, from the deserts to the lush valleys of North-East, forts adorn each and every corner of the landscape of the Indian subcontinent. This section aims to provide a comprehensive overview of these magnificent monuments that bear the stories of the political vicissitudes of our country.)
                
                subcategory 5: Footprint of Freedom
                    Districts Of Defiance(The history of the freedom movement in India comprises a multitude of revolutionary events that helped achieve independence. While a few momentous upheavals and personalities stand out in this historical narrative, the independence of India is also attributed to a series of valuable yet lesser-known incidents that took place in different districts across the country. The Digital District Repository is an attempt to discover and document the memory of these countless stories, events, sites and individuals.)
                    Freedom Archive(This section contains a collection of rare archival material such as books, photographs, gazetteers, letters, newspaper clippings and much more on the freedom struggle of India. The freedom movement engulfed the entire country and people from all walks of life joined hands to drive the foreign oppressors out of this land. Even after more than 7 decades of freedom, these stories of courage, selflessness and determination continue to inspire and instill pride in us. The present section aims to preserve and bring to light rare glimpses of the fight for freedom in the form of digital records.)
                
                subcategory 6: Cultural Chronicles
                    Photo Essays 
                    Retrieved Artefacts Of India(For millennia, India has been a melting pot of diverse cultures, boasting a rich heritage of breathtaking sculptures and artwork. Yet, over centuries, conquerors and colonial powers relentlessly pillaged this heritage, a trend continued by modern looters and smugglers. Consequently, much of India’s historical wealth found its way to Western museums and private collections, resulting in a profound cultural loss that deprives future generations of their rich and intricate heritage. The theft or loss of an artefact signifies the erasure of a piece of history and the collective memory it embodies. Removing artefacts from their original locations strips them of their intrinsic significance, depriving future generations of cultural insights. However, in the past decade, concerted efforts by Indian and international governments, NGOs, journalists, and heritage activists have succeeded in repatriating 358 artefacts back to India. Explore this section to delve into the world of Retrieved Artefacts, uncovering their repatriation stories, heritage, and the legal frameworks that protect them.)
                    Snippets
                    Stories

            2. Textual Repository: Archives, E-Books, Gazettes and Gazetteers, Indian National Bibliography,
            Manuscripts, Other-Collections, Rare Books, Reports and Proceedings, Research Papers, Union Catalogue

            3. Audio & Visual Repository: Audios, Images, Intangible-Cultural-Heritage, Museum-Collections, 
            Paintings, Photos-Archives, Videos

            4. Activities (Games): Crossword, Puzzle, Quiz

        Capabilties: Q/A with the content, Search the website, summarise pages when pages contain lot of text.
        """

    qa_prompt = PromptTemplate.from_template(
        """
            Your name is Bharti. You are an AI assistant for the Indian Culture Portal that deals with Indian Culture and History.

            Instructions:
            - The following is the conversation history so far.
            - Use it only for additional context if needed.
            - Focus on answering ONLY the latest user question.
            - Do not repeat prior answers unless explicitly asked.
            - Do not every start you answer with a greeting.
            - Answer capabilites in pointers.
            - Whenever asked about categories give a brief intro as well.
            - Keep your answer under 100-120 words.

            Context:
            {knowledge_context}

            Conversation History:
            {conversation_history}

            Latest User Question:
            {latest_question}

            Response:
            """
                )

    conversation_history = ""
    for msg in state['user_query']:
        role = (
            "User" if msg.type == "human" else
            "Assistant" if msg.type == "ai" else
            "System"
        )
        conversation_history += f"{role}: {msg.content}\n"

    latest_question = state['user_query'][-1].content if state['user_query'] else ""

    chain = qa_prompt | llm
    response = chain.invoke({
        'conversation_history': conversation_history.strip(),
        'latest_question': latest_question,
        'knowledge_context': knowledge_context
    })

    content = response.content

    if language == 'hi':
        content = translate_to_hindi(content)

    return {'response': content}


# handles specilaised query responses
#------------------------------------------------------------------------------------------------
def specialised_query_answer(state: State):
    language = state['language']

    qa_prompt = PromptTemplate.from_template(
        """
            Your name is Bharti. You are an AI assistant for the Indian Culture Portal that deals with Indian Culture and History.

            Instructions:
            - Answer ONLY using the context provided.
            - Do NOT guess or fabricate anything.
            - Use the conversation history only if relevant.
            - Focus on answering ONLY the latest user question.
            - Group your answer category-wise.
            - Respond ONLY with a valid JSON array, strictly matching this structure:

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

            Context:
            {context}

            Conversation History:
            {conversation_history}

            Latest User Question:
            {latest_question}
        """
    )

    if state['intent'] == "Specialised":
        conversation_history = ""
        if len(state['user_query']) > 1:
            prev_msg = state['user_query'][-2]  
            role = (
                "User" if prev_msg.type == "human" else
                "Assistant" if prev_msg.type == "ai" else
                "System"
            )
            conversation_history = f"{role}: {prev_msg.content}\n"

        latest_question = (
            state['user_query'][-1].content
            if state['user_query']
            else ""
        )

        chain = qa_prompt | llm
        response = chain.invoke({
            'context': state['context'],
            'conversation_history': conversation_history.strip(),
            'latest_question': latest_question
        })

        try:
            parsed_json = json.loads(response.content)
            if not isinstance(parsed_json, list):
                raise ValueError("Expected a list of category dictionaries")
        except Exception as e:
            print(f"Initial Parsing Error: {e}")

            fix_json_prompt = PromptTemplate.from_template(
                """
                    You are a JSON repair tool.
                    Your task is to correct invalid JSON and return only the corrected JSON list.
                    Only fix formatting issues. Just return the correct JSON—do not alter the original content.

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
                    "description": (
                        "Proper response not returned for the query. Try Again!"
                        if language != "Hindi"
                        else "प्रश्न के लिए उचित उत्तर प्राप्त नहीं हुआ। कृपया पुनः प्रयास करें!"
                    ),
                    "resources": []
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


# Answer Queries
#------------------------------------------------------------------------------------------------
def answer_query(state:State):
    if state['intent'] == "Greeting":
        print("Debug: Greeting detected, skipping SQL query.")
        return {'context': ''}  
    
    elif state['intent'] == "Specialised":
        query = state['user_query'][-1]

        similar_title_ids = fetch_similar_titles(vector_store=vector_store, query=query.content) or []

        if similar_title_ids:
            sql_query_result = generate_sql_query(ids=similar_title_ids)
            context = sql_query_result.to_string() if isinstance(sql_query_result, pd.DataFrame) else sql_query_result
            print('context', context)
        else:
            print("Debug: No similar titles found.")
            context = ''

        return {'context': context}
    
    else:
        print("Debug: Unknown intent, returning empty context.")
        return {'context': ''}  


# Langgraph
#------------------------------------------------------------------------------------------------
graph_builder.add_node("question_type", identify_intent)
graph_builder.add_node("context_memory", answer_query)
graph_builder.add_node("greeting_response", greeting_answer)
graph_builder.add_node("general_query_response", general_query_answer)
graph_builder.add_node("specialised_query_response", specialised_query_answer)


graph_builder.add_edge(START,"question_type")
graph_builder.add_edge("question_type", "context_memory")
graph_builder.add_edge("context_memory", 'greeting_response')
graph_builder.add_edge("context_memory", "general_query_response")
graph_builder.add_edge("context_memory", 'specialised_query_response')
graph_builder.add_edge("greeting_response", END)
graph_builder.add_edge("general_query_response", END)
graph_builder.add_edge("specialised_query_response", END)

graph = graph_builder.compile(checkpointer=memory)



@app.post('/chat')
def query():
    global thread_id

    events_list = []
    answer = ''
    
    data = request.get_json()
    user_query = data['query']
    language = data['language']

    config = {"configurable": {"thread_id": thread_id, "language": language}}
    events = graph.stream(
        {
            'user_query': [{'role': 'user', 'content': user_query}],
            'language': language
        },
        config=config
    )
    
    for event in events:
        events_list.append(event)


    if not events_list:
        return jsonify({'answer': 'Cannot generate response. Try Again!'}), 404

    intent = events_list[0]['question_type']['intent']

    # Convenience mapping
    event_map = {}
    for ev in events_list:
        event_map.update(ev)

    if intent == 'Specialised':
        # Specialized query returns JSON
        node = event_map.get('specialised_query_response')
        if node:
            json_data = json.loads(node['response'][0]['content'])
            thread_id += 1

            return jsonify({'answer': json_data}), 200
        else:
            return jsonify({'answer': 'No specialized response generated.'}), 500

    elif intent == 'General':
        node = event_map.get('general_query_response')
        if node:
            return jsonify({'answer': node['response']}), 200
        else:   
            return jsonify({'answer': 'No general response generated.'}), 500

    elif intent == 'Greeting':
        node = event_map.get('greeting_response')
        if node:
            return jsonify({'answer': node['response']}), 200
        else:
            return jsonify({'answer': 'No greeting response generated.'}), 500
        
    else:
        return jsonify({'answer': 'Cannot understand the intent. Please type a proper query.'}), 404


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
    def handle_snippets(parsed_url, page, nid, language):
        try:
            api_url = f'https://icvtesting.nvli.in/rest-v1/snippets?page={page if page != "" else 0}&&field_state_name_value='

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
#------------------------------------------------------------------------------------

    def handle_stories(parsed_url, page, nid, language):
        try:
            api_url = f'https://icvtesting.nvli.in/rest-v1/stories-filter?page={page if page != "" else 0}&&field_state_name_value='

            data = extract_page_content(api_url)
            if not data or 'results' not in data:
                return jsonify({"summary": "No data found"}), 404

            subcategory_data = next((category_data for category_data in data['results'] if str(category_data.get('nid')) == str(nid)), None)

            if subcategory_data:
                answer = summarise_content(clean_and_truncate_html(subcategory_data['field_story_short_descp'], 2000), language)               
                return jsonify({'summary': answer}), 200
            else:
                return jsonify({'summary': 'No NID found to fetch data. Try another page'}), 404
        except Exception as e:
            print(e)
            return jsonify({'summary': 'Failed to summarise the page. Try again!'}), 500 
    
#------------------------------------------------------------------------------------
    
    # Textiles
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
            
            elif (subcategory_type.lower()) == 'textiles-from-museum':
                museum = parsed_url.split('/')[3].split('=')[-1]
                print(museum)
                if museum == 'National-Museum-New-Delhiss' or museum == 'National-Museum-New-Delhi':
                    api_url = 'https://icvtesting.nvli.in/rest-v1/textiles-and-fabrics-of-india/textiles-museum-collections/national-museum?page={page}&&field_state_name_value='
                elif museum == 'Indian-Museum-Kolkata':
                    api_url = 'https://icvtesting.nvli.in/rest-v1/textiles-and-fabrics-of-india/textiles-museum-collections/indian-museum?page={page}&&field_state_name_value='
                elif museum == 'Salar-Jung-Museum-Hyderabad':
                    api_url = 'https://icvtesting.nvli.in/rest-v1/textiles-and-fabrics-of-india/textiles-museum-collections/salarjung-museum?page={page}&&field_state_name_value='
                elif museum == 'Allahabad-Museum-Prayagraj':
                    api_url = 'https://icvtesting.nvli.in/rest-v1/textiles-and-fabrics-of-india/textiles-museum-collections/ald-msm?page={page}&&field_state_name_value='
                elif museum == 'Victoria-Memorial-Hall-Kolkata':
                    api_url = 'https://icvtesting.nvli.in/rest-v1/textiles-and-fabrics-of-india/textiles-museum-collections/vmh?page={0}&&field_state_name_value='
            
            elif (subcategory_type == 'type-of-textiles'):
                section = parsed_url.split('/')[3].lower()
                api_url = f'https://icvtesting.nvli.in/rest-v1/textiles-and-fabrics-of-india/type-of-textile/{section}?page=0&&field_state_name_value='
                subcategory_data = []
                
                print(api_url)
                
                data = extract_page_content(api_url)

                for entry in data['results']:
                    subcategory_data.append({'title': entry['title'], 'state': entry['field_type_state'] , 'description': entry['nothing']})
                
                if subcategory_data:
                    answer = summarise_content(subcategory_data, language)               
                    return jsonify({'summary': answer}), 200
                else:
                    return jsonify({'summary': 'No NID Found to fetch data. Try another page'}), 404


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
    
    # Timelss Trends
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
                print('timelss api called')

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
                subcategory_data = next((category_data for category_data in data['results'] if str(category_data.get('title').lower().strip()) == subcategory_type.replace('-', ' ').lower()), None)

            if subcategory_data:
                answer = summarise_content(subcategory_data, language)               
                return jsonify({'summary': answer}), 200
            else:
                return jsonify({'summary': 'No NID Found to fetch data. Try another page'}), 404

        except Exception as e:
            print(e)
            return jsonify({'summary': 'Failed to summarise the page. Try again!'}), 500

#------------------------------------------------------------------------------------
    
    # Nizams
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
    
    # UNESCO
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
    
    # Forts of India
    def handle_forts(parsed_url, page, nid, language):
        try:
            base_url = 'https://icvtesting.nvli.in/rest-v1/forts-of-india'
            sub_category = parsed_url.split('/')[2].lower()

            if sub_category == 'discover-forts-of-india':
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
                subcategory_data = [category_data['title'] for category_data in data['results'] if 'title' in category_data]
            
            if subcategory_data:
                answer = summarise_content(subcategory_data, language)               
                return jsonify({'summary': answer}), 200
            else:
                return jsonify({'summary': 'No NID Found to fetch data. Try another page'}), 404
            
        except Exception as e:
            print(e)
            return jsonify({'summary': 'Failed to summarise the page. Try again!'}), 500 
#-----------------------------------------------------------------------------

    # Ajanta Caves
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

    # Virtual Walkthrough
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
    
    # DOD
    def handle_DOD(parsed_url, page, nid, language):  
        all_titles = []          
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

            if nid:
                subcategory_data = next((category_data for category_data in data['results'] if str(category_data.get('nid')) == str(nid)), None)
            else:
                for entry in data['results']:
                    all_titles.append(entry['title'])
                subcategory_data = all_titles
            
            if subcategory_data:
                answer = summarise_content(subcategory_data, language)               
                return jsonify({'summary': answer}), 200
            else:
                return jsonify({'summary': 'No NID Found to fetch data. Try another page'}), 404
          
        except Exception as e:
            print(e)
            return jsonify({'summary':'Failed to summarise the page. Try again!'}), 500 
#-----------------------------------------------------------------------------

    # Artifacts
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

    # Freedom Fighters
    def handle_freedom_fighters(parsed_url, page, nid, language):
        try:
            base_url = 'https://icvtesting.nvli.in/rest-v1/freedom-archive' 
            sub_category = parsed_url.split('/')[2].lower()

            if sub_category == 'unsung-heroes':
                api_url = f'https://icvtesting.nvli.in/rest-v1/unsung-heroes?page={page if page != "" else 0}&&field_state_name_value='
            if sub_category == 'historic-cities':
                city = parsed_url.split('/')[3].lower().strip()
                if city != 'patna':
                    api_url = f'https://icvtesting.nvli.in/rest-v1/{city}/Historic_Cities_Freedom_Movement?page=0&&field_state_name_value='
                else:
                    api_url = 'https://icvtesting.nvli.in/rest-v1/historic-cities/patna/Historic-cities-freedom-movement?page=0&&field_state_name_value='
            if sub_category == 'forts':
                api_url = f'https://icvtesting.nvli.in/rest-v1/forts-of-india/forts-and-freedom-struggle?page=0&&field_state_name_value='
            if sub_category == 'textile':
                api_url = f'https://icvtesting.nvli.in/rest-v1/INDIGO-DYE-AND-REVOLT?page=0&&field_state_name_value='
        
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
    
    # Food and Culture
    def handle_food_and_culture(parsed_url, page, nid, language):
        try:
            sub_category = parsed_url.split('/')[2].lower()

            # If "cuisines-of-india"
            if sub_category == 'cuisines-of-india':
                # Static data for Cuisines of India
                subcategory_data = [
                        "This page contains information about cusines of different greographic regions of India. Such as Rajasthan in West. Bihar in the East. Kerala and Andhra in the South. Maharashtra in the Central India. Apart from thatb it also contains info about cusines from various North Eastern States. "
                    
                ]

                if not subcategory_data:
                    return jsonify({"summary": "No data found"}), 404

                # Summarise the static data
                answer = summarise_content(subcategory_data, language)
                return jsonify({"summary": answer}), 200

            # If "royal-table"
            if sub_category == 'royal-table':
                api_url = 'https://icvtesting.nvli.in/rest-v1/cuisine-royal-table?page=0&field_state_name_value='
                data = extract_page_content(api_url)

                if not data or 'results' not in data:
                    return jsonify({"summary": "No data found"}), 404

                # Get the matching NID if specified
                if nid:
                    subcategory_data = next(
                        (entry for entry in data['results'] if str(entry.get('nid')) == str(nid)),
                        None
                    )
                    if not subcategory_data:
                        return jsonify({'summary': 'No NID Found to fetch data. Try another page'}), 404
                else:
                    subcategory_data = data['results']

                # Summarise
                answer = summarise_content(subcategory_data, language)
                return jsonify({'summary': answer}), 200
            
            if sub_category == 'stirring-through-time':
                api_url = 'https://icvtesting.nvli.in/rest-v1/evolution-cuisine?page=0&&field_state_name_value='
                data = extract_page_content(api_url)

                if not data or 'results' not in data:
                    return jsonify({"summary": "No data found"}), 404

                # Get the matching NID if specified
                if nid:
                    subcategory_data = next(
                        (entry for entry in data['results'] if str(entry.get('nid')) == str(nid)),
                        None
                    )
                    if not subcategory_data:
                        return jsonify({'summary': 'No NID Found to fetch data. Try another page'}), 404
                else:
                    subcategory_data = data['results']

                # Summarise
                answer = summarise_content(subcategory_data, language)
                return jsonify({'summary': answer}), 200


            # If none of the known subcategories matched
            return jsonify({'summary': 'Unknown subcategory.'}), 400

        except Exception as e:
            print("Error in handle_food_and_culture:", e)
            return jsonify({'summary': 'Failed to summarise the page. Try again!'}), 500

#-----------------------------------------------------------------------------

    # Festivals of India        
    def handle_festivals(parsed_url, page, nid, language):
        try:
            base_url = 'https://icvtesting.nvli.in/rest-v1/festivals-of-india' 
            category = parsed_url.split('/')[2].lower()
            sub_category = parsed_url.split('/')[3].lower()

            if category == 'pan-indian-festivals':
                api_url = 'https://icvtesting.nvli.in/rest-v1/festivals-of-India/pan-indian-festivals?page=0&&field_state_name_value='
            if category == 'fairs-and-pilgrimages':
                if sub_category not in ['fairs', 'pilgrimages']:
                    api_url = f'https://icvtesting.nvli.in/rest-v1/festivals-of-India/fairs-pilgrimages/fairs?page=0&&field_state_name_value='
                if sub_category == 'fairs':
                    api_url = f'https://icvtesting.nvli.in/rest-v1/festivals-of-India/fairs-pilgrimages/fairs?page=0&&field_state_name_value='
                elif sub_category == 'pilgrimages':
                    api_url = 'https://icvtesting.nvli.in/rest-v1/festivals-of-India/fairs-pilgrimages/pilgrimage?page=0&&field_state_name_value='
            if category == 'folk-festivals':
                if sub_category not in ['honoring-deities','social-traditions', 'celebrating-nature']:
                    api_url = 'https://icvtesting.nvli.in/rest-v1/festivals-of-India/folk-festivals/Celebrating-Nature?page=0&&field_state_name_value='
                if sub_category == 'honoring-deities':
                    api_url = 'https://icvtesting.nvli.in/rest-v1/festivals-of-India/folk-festivals/Honouring-Deities?page=0&&field_state_name_value='
                if sub_category == 'social-traditions':
                    api_url = 'https://icvtesting.nvli.in/rest-v1/festivals-of-India/folk-festivals/Social-Traditions?page=0&&field_state_name_value='
                if sub_category == 'celebrating-nature':
                    api_url = 'https://icvtesting.nvli.in/rest-v1/festivals-of-India/folk-festivals/Celebrating-Nature?page=0&&field_state_name_value='
            if category == 'tribal-festivals':
                if sub_category not in ['venerating-ancestors-and-deities', 'worshipping-nature']:
                    api_url = 'https://icvtesting.nvli.in/rest-v1/festivals-of-India/tribal-festivals/worshipping-nature?page=0&&field_state_name_value='
                if sub_category == 'venerating-ancestors-and-deities':
                    api_url = f'https://icvtesting.nvli.in/rest-v1/festivals-of-India/tribal-festivals/venerating-ancestors-deities?page=0&&field_state_name_value='
                elif sub_category == 'worshipping-nature':
                    api_url = 'https://icvtesting.nvli.in/rest-v1/festivals-of-India/tribal-festivals/worshipping-nature?page=0&&field_state_name_value='
           
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
    
    # States of India
    def handle_states(parsed_url, page, nid, language):
        try:
            category = parsed_url.split('/')[2].lower().strip()
            sub_category = parsed_url.split('/')[3].lower().strip()
            section = parsed_url.split('/')[4].lower().strip()

            if category == 'north-east-india':
                split = parsed_url.split('/')[5].lower().strip()
                if sub_category == 'unsung-heroes':
                    api_url = f'https://icvtesting.nvli.in/rest-v1/north-east-archive/unsung-heroes?page=0&&field_state_name_value=?page={page if page != "" else 0}&&field_state_name_value='
                if sub_category == 'capital-cities-north-east-india':
                    if section == 'shillong':
                        if split == 'history-evolution':
                            api_url = f'https://icvtesting.nvli.in/rest-v1/north-east-marker-shillong-history?page=0&&field_state_name_value='
                        if split == 'natural-built-heritage':
                            api_url = f'https://icvtesting.nvli.in/rest-v1/north-east-marker-shillong-natural-and-built?page=0&&field_state_name_value='
                        if split == 'streets-localities':
                            api_url = f'https://icvtesting.nvli.in/rest-v1/north-east-marker-shillong-street-and-localities?page=0&&field_state_name_value='
                        if split == 'life-in-the-city':
                            api_url = f'https://icvtesting.nvli.in/rest-v1/north-east-marker-shillong-life-in-the-city?page=0&&field_state_name_value='

                        content = {}

                        data = extract_page_content(api_url)

                        if not data or 'results' not in data:
                            return jsonify({"summary": "No data found"}), 404

                        if nid:
                            subcategory_data = next((category_data for category_data in data['results'] if str(category_data.get('nid')) == str(nid)), None)
                        else:
                            subcategory_data = data
                    
                            for res in subcategory_data['results']:
                                first_100_words = ' '.join(clean_html(res['body']).split()[:30])
                                content[res['title']] = first_100_words

                            if content:
                                answer = summarise_content(content, language) 
                                return jsonify({'summary': answer}), 200 

                elif sub_category == 'tales-from-the-hinterland':
                    api_url = f'https://icvtesting.nvli.in/rest-v1/tales-from-the-hinterland?page={page if page != "" else 0}&&field_state_name_value='

            if category == 'east-india':
                if sub_category == 'bihar':
                    if section == 'tidbits-tales-and-trivia':
                        api_url = f'https://icvtesting.nvli.in/rest-v1/states-of-india/bihar/tidbits-tales-trivia?page=0&&field_state_name_value='
                    if section == 'digital-archives':
                        api_url = 'https://icvtesting.nvli.in/rest-v1/states-of-india/bihar/digital-archives?page=0&&field_state_name_value='
                    elif section == 'bihar-through-traveller-s-gaze':
                        api_url = 'https://icvtesting.nvli.in/rest-v1/states-of-india/bihar/bihar-through?page=0&&field_state_name_value='
                        
                        content = {}
                        data = extract_page_content(api_url)

                        if not data or 'results' not in data:
                            return jsonify({"summary": "No data found"}), 404

                        if nid:
                            subcategory_data = next((category_data for category_data in data['results'] if str(category_data.get('nid')) == str(nid)), None)
                        else:
                            subcategory_data = data
                    
                            for res in subcategory_data['results']:
                                first_100_words = ' '.join(clean_html(res['body']).split()[:80])
                                content[res['title']] = first_100_words


                            if content:
                                answer = summarise_content(content, language) 
                                return jsonify({'summary': answer}), 200 

                    elif section == 'art-and-architecture':
                        api_url = 'https://icvtesting.nvli.in/rest-v1/art-and-architecture-api-pins?page=0&&field_state_name_value='
                        
                        content = {}
                        data = extract_page_content(api_url)

                        if not data or 'results' not in data:
                            return jsonify({"summary": "No data found"}), 404

                        if nid:
                            subcategory_data = next(
                                (category_data for category_data in data['results'] if str(category_data.get('nid')) == str(nid)),
                                None)
                        else:
                            subcategory_data = data

                            for res in subcategory_data['results']:
                                    try:
                                        title = res.get('title', 'Untitled')
                                        raw_json = res.get('field_marker_details', '{}')
                                        parsed_json = json.loads(raw_json)
                                        search_results = parsed_json.get('search_results', [])
                                        if not search_results or not isinstance(search_results, list):
                                            continue 

                                        marker = search_results[0]
                                        html_description = marker.get('field_marker_description', '').strip()

                                        if not html_description:
                                            continue  

                                        soup = BeautifulSoup(html_description, 'html.parser')
                                        clean_text = soup.get_text(separator=' ', strip=True)

                                        first_80_words = ' '.join(clean_text.split()[:80])

                                        if first_80_words:
                                            content[title] = first_80_words

                                    except Exception as e:
                                        print(f"Error processing {res.get('title', 'Unknown')}: {e}")
                                        continue

                            if content:
                                answer = summarise_content(content, language)
                                return jsonify({'summary': answer}), 200
                            else:
                                return jsonify({'summary': 'No valid descriptions found.'}), 404
                                            
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
            api_url = f'https://icvtesting.nvli.in/rest-v1/iconic-battle-of-india/detail?page=&&field_state_name_value='

            data = extract_page_content(api_url)
            if not data or 'results' not in data:
                return jsonify({"summary": "No data found"}), 404

            # Find the record with matching nid
            subcategory_data = next(
                (category_data for category_data in data['results'] if str(category_data.get('nid')) == str(nid)),
                None
            )

            if not subcategory_data:
                return jsonify({'summary': 'No NID found to fetch data. Try another page'}), 404

            desc = subcategory_data.get('field_iconic_marker', '').strip()
            if not desc:
                return jsonify({'summary': 'No detailed content found.'}), 404

            try:
                nested_json = json.loads(f"[{desc}]")
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                return jsonify({'summary': 'Failed to parse detailed content.'}), 500

            target_titles = {
                "Background",
                "Road to Battle",
                "Technology and Army composition ",
                "The Battle",
                "Aftermath"
            }

            summaries = []

            for section in nested_json:
                search_results = section.get("search_results", [])
                for item in search_results:
                    section_title = item.get("title", "").strip()
                    if section_title in target_titles:
                        raw_html = item.get("body", "")
                        clean_text = clean_html(raw_html)

                        # Limit to 100 words
                        words = clean_text.split()
                        limited_text = ' '.join(words[:100]) + ('...' if len(words) > 100 else '')

                        summaries.append(f"**{section_title}**\n\n{limited_text}")

            if not summaries:
                return jsonify({'summary': 'No relevant sections found.'}), 404

            answer = "\n\n".join(summaries)

            return jsonify({'summary': answer}), 200

        except Exception as e:
            print(e)
            return jsonify({'summary': 'Failed to summarise the page. Try again!'}), 500
 
#-----------------------------------------------------------------------------

    # lengendary figures
    def handle_legendary_figures(parsed_url, page, nid, language):
        try:
            section = parsed_url.split('/')[2].lower().strip()
            if section == 'kings-and-queens':
                api_url = 'https://icvtesting.nvli.in/rest-v1/legendary-figure/kings-queens?page=0&&field_state_name_value='
            if section == 'social-reformers-and-revolutionaries':
                api_url = f'https://icvtesting.nvli.in/rest-v1/legendary-figure/social-reformers-revolutionaries?page=0&&field_state_name_value='
            if section == 'sages-philosophers-and-thinkers':
                api_url = 'https://icvtesting.nvli.in/rest-v1/legendary-figure/sages-philosophers-thinkers?page=0&&field_state_name_value='

            data = extract_page_content(api_url)
            if not data or 'results' not in data:
                return jsonify({"summary": "No data found"}), 404

            subcategory_data = next((category_data for category_data in data['results'] if str(category_data.get('nid')) == str(nid)), None)
            
            # print(subcategory_data)
            if subcategory_data:
                answer = summarise_content(remove_src_attributes(subcategory_data['body']), language)               
                return jsonify({'summary': answer}), 200
            else:
                return jsonify({'summary': 'No NID found to fetch data. Try another page'}), 404
        except Exception as e:
            print(e)
            return jsonify({'summary': 'Failed to summarise the page. Try again!'}), 500 
#-----------------------------------------------------------------------------
    
    # healing through ages
    def handle_healing_through_the_ages(parsed_url, page, nid, language):
        sub_category = parsed_url.split('/')[2].lower().strip()

        try: 
            if sub_category == 'pan-india-traditions':   
                api_url = f'https://icvtesting.nvli.in/rest-v1/healing-through-the-ages/pan-india-traditions?page={page if page != "" else 0}&&field_state_name_value='

                data = extract_page_content(api_url)

                if not data or 'results' not in data:
                    return jsonify({"summary": "No data found"}), 404

                sub_data = next((category_data for category_data in data['results'] if str(category_data.get('nid')) == str(nid)), None)
                history = sub_data['body']
                philosophy =  sub_data['field_philosophy']
                practitioners =  sub_data['field_practitioners']
                literature =  sub_data['field_literature']
                surgical_equipment =  sub_data['field_surgical_equipment']

                subcategory_data = f'''history: {history}.\n Philosophy: {philosophy}.\n Practitioners: {practitioners}
                                        Literature: {literature}.\n Surgical Equipment: {surgical_equipment}   
                                    '''

                if subcategory_data:
                    answer = summarise_content(subcategory_data, language)               
                    return jsonify({'summary': answer}), 200
                else:
                    return jsonify({'summary': 'No NID found to fetch data. Try another page'}), 404

            if sub_category == 'unconventional-traditions':
                api_url = f'https://icvtesting.nvli.in/rest-v1/healing-through-the-ages/unconventional-traditions?page=0&&field_state_name_value='

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
            api_url = f'https://icvtesting.nvli.in/rest-v1/classical-dances-details?page={page if page != "" else 0}&field_state_name_value='
            print('api_url', api_url)
            data = extract_page_content(api_url)
            if not data or 'results' not in data:
                return jsonify({"summary": "No data found"}), 404

            # Find the matching item by nid
            subcategory_data = next(
                (category_data for category_data in data['results'] if str(category_data.get('nid')) == str(nid)),
                None
            )

            if not subcategory_data:
                return jsonify({'summary': 'No NID found to fetch data. Try another page'}), 404

            body_html = subcategory_data.get('body', '')
            main_text = clean_html_truncate(body_html, max_words=250)

            other_paragraphs = []
            field_popup = subcategory_data.get('field_popup', '')
            if field_popup:
                try:
                    # Wrap the concatenated JSON objects into an array to make valid JSON
                    json_text = f"[{field_popup.strip()}]"
                    popup_json_array = json.loads(json_text)

                    # Process each object in the array
                    for popup_json in popup_json_array:
                        results = popup_json.get("search_results", [])
                        for res in results:
                            para_html = res.get("field_paragraph", "")
                            if para_html:
                                cleaned_para = clean_html_truncate(para_html, max_words=100)
                                other_paragraphs.append(cleaned_para)
                except Exception as e:
                    print("Error parsing field_popup JSON:", e)

            # Combine main text and other paragraphs
            combined_text = main_text
            if other_paragraphs:
                combined_text += "\n\n" + "\n\n".join(other_paragraphs)

            text_data = {"body": combined_text}
            answer = summarise_content(text_data, language)
            return jsonify({'summary': answer}), 200

        except Exception as e:
            print("Error in classical dances handler:", e)
            return jsonify({'summary': 'Failed to summarise the page. Try again!'}), 500

#-----------------------------------------------------------------------------

    def handle_historical_cities(parsed_url, page, nid, language):
        try:
            city = parsed_url.split('/')[2].lower().strip()
            api_url = f'https://icvtesting.nvli.in/rest-v1/historic-cities?page=0&&field_state_name_value='
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
#-----------------------------------------------------------------------------

    def handle_historic_cities_freedom_movement(parsed_url, page, nid, language):
        try:
            city = parsed_url.split('/')[2].lower().strip()
            if city != 'patna':
                api_url = f'https://icvtesting.nvli.in/rest-v1/{city}/Historic_Cities_Freedom_Movement?page=0&&field_state_name_value='
            else:
                api_url = 'https://icvtesting.nvli.in/rest-v1/historic-cities/patna/Historic-cities-freedom-movement?page=0&&field_state_name_value='

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

    def handle_historic_cities(parsed_url, page, nid, language):
        print('func called')
        try:
            api_url = 'https://icvtesting.nvli.in/rest-v1/historic-cities/delhi/Historic_Cities_Freedom_Movement?page=0&&field_state_name_value='
        
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

    def handle_folktales(parsed_url, page, nid, language):
        try:
            sub_category = parsed_url.split('/')[2].lower().strip()
            if sub_category == 'fables':
                section = parsed_url.split('/')[-1].lower().strip().replace('-', '_')
                api_url = f'https://icvtesting.nvli.in/rest-v1/folktales-of-india/fables?Fables_type={section}'
                print('debug', api_url)
            if sub_category == 'fairytales':
                api_url = 'https://icvtesting.nvli.in/rest-v1/fairytales-landing-main?page=0&&field_state_name_value='
            if sub_category == 'legends':
                api_url = 'https://icvtesting.nvli.in/rest-v1/folktales-of-india/legends?page=0&&field_state_name_'

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

    def handle_lengendary_figures(parsed_url, page, nid, language):
        try:
            sub_category = parsed_url.split('/')[2].lower().strip()

            if sub_category == 'kings-and-queens':
                api_url = 'https://icvtesting.nvli.in/rest-v1/legendary-figure/kings-queens?page=0&&field_state_name_value='
            
            if sub_category == 'sages-philosophers-and-thinkers':
                api_url = 'https://icvtesting.nvli.in/rest-v1/legendary-figure/social-reformers-revolutionaries?page=0&&field_state_name_value='
            
            if sub_category == 'social-reformers-and-revolutionaries':
                api_url = 'https://icvtesting.nvli.in/rest-v1/legendary-figure/sages-philosophers-thinkers?page=0&&field_state_name_value='

            data = extract_page_content(api_url)
            if not data or 'results' not in data:
                return jsonify({"summary": "No data found"}), 404

            subcategory_data = next((category_data for category_data in data['results'] if str(category_data.get('nid')) == str(nid)), None)

            if subcategory_data:
                answer = summarise_content(clean_and_truncate_html(subcategory_data['body']), language)               
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

        print('category', category)

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
        elif category == 'digital-district-repository':
            return handle_DOD(parsed_url, page, nid, language=lang)
        elif category == 'retrieved-artefacts-of-india':
            return handle_artifacts(parsed_url, page, nid, language=lang)
        elif category == 'freedom-archive':
            return handle_freedom_fighters(parsed_url, page, nid, language=lang)
        elif category == 'food-and-culture':
            return handle_food_and_culture(parsed_url, page, nid, language=lang)
        elif category == 'festivals-of-india':
            return handle_festivals(parsed_url, page, nid, language=lang)
        elif category == 'states-of-india':
            return handle_states(parsed_url, page, nid, language=lang)
        elif category == 'iconic-battles-of-india':
            return handle_iconic_battles(parsed_url, page, nid, language=lang)
        elif category == 'legendary-figures-of-india':
            return handle_legendary_figures(parsed_url, page, nid, language=lang)
        elif category == 'healing-through-the-ages':
            return handle_healing_through_the_ages(parsed_url, page, nid, language=lang)
        elif category == 'classical-dances-of-india':
            return handle_classical_dances(parsed_url, page, nid, language=lang)
        elif category == 'historic-cities-of-india':
            return handle_historical_cities(parsed_url, page, nid, language=lang)
        elif category == 'historic-cities-freedom-movement':
            return handle_historic_cities_freedom_movement(parsed_url, page, nid, language=lang)
        elif category ==  'historic-cities':
            return handle_historic_cities(parsed_url, page, nid, language=lang )
        elif category == 'snippets':
            return handle_snippets(parsed_url, page, nid, language=lang)
        elif category == 'stories':
            return handle_stories(parsed_url, page, nid, language=lang)
        elif category == 'folktales-of-india':
            return handle_folktales(parsed_url, page, nid, language=lang)
        elif category == 'legendary-figures-of-india':
            return handle_lengendary_figures(parsed_url, page, nid, language=lang)
    except Exception as e:
            print(e)

#------------------------------------------------------------------------------------
@app.get('/clear_memory')
def clear_memory():
    global thread_id
    thread_id += 1
    return jsonify({"message": "Memory cleared successfully"}), 200
