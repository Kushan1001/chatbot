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

CORS(app)

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
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            data = response.json()
            return data
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None
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
            - Includes administrative questions about the portal itself and categories description.(e.g., "Who developed this portal?", "What can you do?", "What is NVLI", "Url of a particular category", "What is a particular category about", "What is folktales about", What is timeless trends category about", "Description of ebooks").
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
        The content is available both in 18 languages.

        
        Categories available on the website:
            1. Original Catgegories (Prepared with original research by reserachers done here):
                subcategory 1: Cultural Expressions
                List of them:
                    Classical Dances of Indian (India's classical dances are a vibrant expression of its 
                    diverse cultural heritage, deeply rooted in tradition, mythology, and spiritual practice. 
                    Each form tells stories through graceful movements, intricate rhythms, and expressive gestures. 
                    These dance forms not only reflect regional identities but also embody centuries-old philosophies 
                    and artistic disciplines. This category explores the origins, evolution, and unique characteristics of 
                    India's classical dance styles, offering a window into the artistic soul of the nation.)
                    (url = https://icpdelhi.nvli.in/classical-dances-of-india),
                    
                    Festivals Of India (To experience the festivals of India is to experience the grandeur 
                    and richness of the Indian cultural heritage. The festivals of India thrive in a culture 
                    of diversity, and the celebration of these festivals has become a time for cross-cultural 
                    exchanges. Filled with rituals, music, performances, culinary treats, and more, each 
                    festival presents its own fascinating history and unique charm. A large diversity of 
                    customs, traditions, and tales are also associated with festivals. Learn about the 
                    cultural diversity, customs and traditions, as well as the fascinating stories 
                    associated with the festivals presented in the categories below, or explore the vibrant 
                    festivals of the states by clicking on the map or finding your favourite festival.)
                    (url = https://icpdelhi.nvli.in/festivals-of-india), 
                    
                    Food And Culture(The Indian culinary repertoire reflects the cultural diversity of the 
                    country. The term “Indian food” denotes a mélange of flavours from different parts of 
                    the country and showcases centuries of cultural exchange with the far corners of the 
                    world. Here, on our portal, we are making a small effort of gradually building a 
                    treasure trove of information about the countless exquisite flavours of our country. 
                    It is an ongoing venture and over time we aim to capture as much as possible of the 
                    incredible culinary diversity of this land.), Musical Instruments Of India
                    (url = https://icpdelhi.nvli.in/food-and-culture),

                    Musical Instruments of India(The Musical Instruments section of the Indian Culture 
                    portal contains information about a range of instruments from across India. 
                    The Indian Culture portal has researched and is happy to present information 
                    about the countless exquisite musical instruments of our country.)
                    (url = https://icpdelhi.nvli.in/musical-instruments-of-india),

                    Textiles and fabrics of India (Textiles and Fabrics of India is an attempt to showcase 
                    and celebrate the long and diverse tradition of Textiles in India. The history of this 
                    craft goes back to the ancient period. This section highlights and honours the 
                    craftsmanship of the Indian handloom workers, embroiderers, block printers, painters 
                    and others who have immensely contributed to build a distinct textile industry for India.)
                    (url = https://icpdelhi.nvli.in/textiles-and-fabrics-of-india),

                    Timeless Trends (n both its traditional and modern manifestations, Indian art exhibits a 
                    powerful sense of design and a vivid imagination. These are reflected in sculptures, 
                    paintings, murals, architecture, coins, and items of personal adornment like jewellery, 
                    clothing, and more. Surviving the vagaries of time, many of these artefacts are now 
                    preserved in museums, archaeological sites and cultural institutions. These seemingly 
                    ordinary artefacts act as a repository of knowledge that conveys information about the 
                    society of their time. Timeless Trends celebrates the interconnectedness of the past and 
                    the present and attempts to discover the links between the cultures and traditions we 
                    cherish, the structures and sites that dot our modern landscapes, and the little things 
                    that we do and say every day.) (url = https://icpdelhi.nvli.in/timeless-trends),
                    
                subcategory 2: Legends and Legacies
                    Folktales Of India (India has a rich and diverse tradition of folktales, shaped by its many languages, 
                    cultures, and regions. These stories—ranging from fables and fairytales to myths and legends—have been 
                    passed down through generations, reflecting the values, beliefs, and imaginations of the people. 
                    Fables use animal characters to teach moral lessons, while fairytales often involve magical beings, 
                    heroic quests, and transformations. Myths recount the exploits of gods and divine beings, explaining 
                    creation, duty, and cosmic order, whereas legends celebrate the lives of saints, warriors, poets, and 
                    jesters whose deeds live on in collective memory. Whether shared in gatherings, temples, courts, or 
                    classrooms, these tales continue to captivate audiences, preserving the spirit of India's vibrant 
                    storytelling heritage.) (url = https://icpdelhi.nvli.in/folktales-of-india),
                    
                    Healing Through The Ages(The 'Healing Through the Ages' category aims to trace the various dimensions and understanding of ailments and cures across India. It is a repository 
                    which brings together the different meanings of 'health' and provides an overview of both conventional 
                    and unconventional approaches to maintaining balance and restoring a sense of well-being. It will help 
                    you traverse historical, regional and cultural boundaries, and help you to cultivate a nuanced 
                    understanding of suffering and healing.) (url = https://icpdelhi.nvli.in/healing-through-the-ages),

                    Iconic Battles Of India(Warfare has shaped the course of Indian history. The subcontinent 
                    has witnessed epic battles that not only altered its destiny but also influenced the world 
                    at large. This section delves into twelve iconic battles that changed the tide of Indian 
                    history, tracing the evolution of warfare across different eras. Each of these conflicts 
                    marked the rise or fall of dynasties, introduced new systems of governance, and gave birth 
                    to lasting traditions, beliefs, and cultural patterns that continue to shape India's 
                    identity today.) (url = https://icpdelhi.nvli.in/iconic-battles-of-india),
                     
                    Jewellery Of The Nizams (url = https://icpdelhi.nvli.in/jewellery-of-the-nizams),
                    
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
                    (url = https://icpdelhi.nvli.in/legendary-figures-of-india),
                
                subcategory 3: Pan India Explorations
                    Historic Cities Of India(The map of India is dotted with cities that so many of us call home. Many of these cities have origins in our collective history. While they may now be modern and dynamic centres, they continue to represent centuries of culture and heritage that even today, sets them apart from every other city across the globe. Explore these unique urban centres and everything that they have to offer at your own pace, through a virtual expedition. Click on the icons to the right to begin a virtual visit to these historic cities! Each city has its own story, one that is told here through a collection of rare photographs, multimedia, specially-narrated tales, and more. We invite you to sift through them, explore, and discover your own favourite stories about every city.) (url = 'https://icpdelhi.nvli.in/historic-cities-of-india'),
                    States Of India(India, a vast and vibrant nation, is a mosaic of diverse states, each woven with a unique thread of culture, history, and tradition. From the majestic, snow-capped mountains in the north to the sun-drenched coastlines in the south, each state offers a rich and varied blend of languages, cuisines, arts, and festivals. This category delves into the architectural marvels that adorn the country, the profound literary contributions from various corners, and the abundant interesting anecdotes that shape each state's identity. It also meticulously charts the historical development of these regions through different ages, reflecting India's millennia- old heritage. Each state, with its distinct character, forms a thread in the extraordinary fabric of India.) (url = 'https://icpdelhi.nvli.in/states-of-india'),
                    Unesco (url = https://icpdelhi.nvli.in/unesco),
                
                subcategory 4: Built Heritage
                    3d Explorations(The Indian culinary repertoire reflects the cultural diversity of the country. The term “Indian food” denotes a mélange of flavours from different parts of the country and showcases centuries of cultural exchange with the far corners of the world. Here, on our portal, we are making a small effort of gradually building a treasure trove of information about the countless exquisite flavours of our country. It is an ongoing venture and over time we aim to capture as much as possible of the incredible culinary diversity of this land.) (url = https://icpdelhi.nvli.in/3d-Explorations),
                    Ajanta Caves(The Ajantā caves are rock-cut Buddhist cave temples carved out of a horseshoe shaped valley near the Waghora river at the edge of the Indyadhri range. The caves are a UNESCO World Heritage site and are thronged by thousands of tourists who come to admire its serene location, rock-cut architecture and beautiful Buddhist paintings that are found in the caves. These 30 rock-cut caves are part of a constellation of Buddhist cave temples dotting the Sahayādri or Western Ghats in Maharashtra. But Ajantā is unique as it hosts the finest specimens of art - Cave 9 and 10 contain the oldest Buddhist narrative paintings in India.) (url = https://icpdelhi.nvli.in/3d-Explorations),
                    Forts Of India(The Forts of India are some of the most awe-inspiring monuments found in the country. From the Himalayas to the peninsular tip, from the deserts to the lush valleys of North-East, forts adorn each and every corner of the landscape of the Indian subcontinent. This section aims to provide a comprehensive overview of these magnificent monuments that bear the stories of the political vicissitudes of our country.) (url = https://icpdelhi.nvli.in/forts-of-india),
                
                subcategory 5: Footprint of Freedom
                    Districts Of Defiance(The history of the freedom movement in India comprises a multitude of revolutionary events that helped achieve independence. While a few momentous upheavals and personalities stand out in this historical narrative, the independence of India is also attributed to a series of valuable yet lesser-known incidents that took place in different districts across the country. The Digital District Repository is an attempt to discover and document the memory of these countless stories, events, sites and individuals.) (url = https://icpdelhi.nvli.in/digital-district-repository),
                    Freedom Archive(This section contains a collection of rare archival material such as books, photographs, gazetteers, letters, newspaper clippings and much more on the freedom struggle of India. The freedom movement engulfed the entire country and people from all walks of life joined hands to drive the foreign oppressors out of this land. Even after more than 7 decades of freedom, these stories of courage, selflessness and determination continue to inspire and instill pride in us. The present section aims to preserve and bring to light rare glimpses of the fight for freedom in the form of digital records.) (url = https://icpdelhi.nvli.in/freedom-archive),
                
                subcategory 6: Cultural Chronicles
                    Photo Essays (url = https://icpdelhi.nvli.in/photo-essays),
                    Retrieved Artefacts Of India(For millennia, India has been a melting pot of diverse cultures, boasting a rich heritage of breathtaking sculptures and artwork. Yet, over centuries, conquerors and colonial powers relentlessly pillaged this heritage, a trend continued by modern looters and smugglers. Consequently, much of India's historical wealth found its way to Western museums and private collections, resulting in a profound cultural loss that deprives future generations of their rich and intricate heritage. The theft or loss of an artefact signifies the erasure of a piece of history and the collective memory it embodies. Removing artefacts from their original locations strips them of their intrinsic significance, depriving future generations of cultural insights. However, in the past decade, concerted efforts by Indian and international governments, NGOs, journalists, and heritage activists have succeeded in repatriating 358 artefacts back to India. Explore this section to delve into the world of Retrieved Artefacts, uncovering their repatriation stories, heritage, and the legal frameworks that protect them.) (url = https://icpdelhi.nvli.in/retrieved-artefacts-of-india),
                    Snippets (url = https://icpdelhi.nvli.in/retrieved-artefacts-of-india),
                    Stories (url = https://icpdelhi.nvli.in/stories),

            2. Textual Repository: Archives (url = https://icpdelhi.nvli.in/archives), E-Books (url = https://icpdelhi.nvli.in/e-books), Gazettes and Gazetteers (url = https://icpdelhi.nvli.in/gazettes-and-gazetteers), Indian National Bibliography (url = https://inb.nvli.in/cgi-bin/koha/opac-search.pl?advsearch=1&idx=kw&limit=branch%3ACRL&sort_by=relevance&do=Search),
            Manuscripts (url = https://icpdelhi.nvli.in/manuscripts), Other-Collections (url = https://icpdelhi.nvli.in/other-collections), Rare Books (url = https://icpdelhi.nvli.in/rare-books), Reports and Proceedings (url = https://icpdelhi.nvli.in/reports-and-proceedings) , Research Papers (url = https://icpdelhi.nvli.in/research-papers), Union Catalogue (url = https://indianculture.gov.in/union-catalogue),

            3. Audio & Visual Repository: Audios (url = https://icpdelhi.nvli.in/audios), Images (url = https://icpdelhi.nvli.in/images), Intangible-Cultural-Heritage (url = https://icpdelhi.nvli.in/intangible-cultural-heritage), Museum-Collections (url = https://icpdelhi.nvli.in/museum-collections), 
            Paintings(url = https://icpdelhi.nvli.in/paintings), Photos-Archives (url = https://icpdelhi.nvli.in/photo-archives), Videos (url = https://icpdelhi.nvli.in/videos),

            4. Activities (Games): Crossword (url = https://icpdelhi.nvli.in/Crossword), Puzzle (url = https://icpdelhi.nvli.in/Puzzle), Quiz (url = https://icpdelhi.nvli.in/Quiz),

            5. Tools: Flagship Events: The Indian Culture Portal (ICP) was an integral participant in five flagship events organized by the Ministry of Culture. Beyond its interactive and informative exhibition booths, ICP played a pivotal role in the conceptualization, design, and production of key publications and launches. Through thoughtfully curated content, immersive visitor experiences, and innovative communication tools, ICP contributes significantly to the dissemination and celebration of India's cultural heritage (url = https://icpdelhi.nvli.in/flagship-events),
                      Outreach: Discover how the Indian Culture Portal connects with institutions, communities, and experts across the country to preserve and promote India’s rich cultural heritage. Our outreach initiatives foster collaboration, build awareness, and bring culture closer to people through workshops, partnerships, and public engagement (url = https://icpdelhi.nvli.in/outreach),
                      Publications: This section showcases original publications, books, and graphic novels developed by the Indian Culture Portal, alongside publications created by partner organizations within the Ministry of Culture. It also includes select works that were officially unveiled during key Ministry events (url = https://icpdelhi.nvli.in/publications),


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
            - Give the url of category when asked about a prticular category. Urls should be hyperlinks. Hyperlink name should be same as the category name. Do not give them as texts.
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
        print('here')
        try:
            print(nid)
            path_parts = [seg for seg in urlparse(parsed_url).path.split('/') if seg]
            sub_category = (path_parts[0].lower() if path_parts else '')
            instrument_slug = (path_parts[1] if len(path_parts) > 1 else '').strip()
            print(sub_category)

            base_url = 'https://icvtesting.nvli.in/rest-v1/musical-instruments'

            # paging strategy same as your first function
            pages_to_try = [page] if page != "" else list(range(0, 5))
            if page != "" and page not in pages_to_try:
                pages_to_try.extend(range(0, 6))

            for p in pages_to_try:
                api_url = f'{base_url}/{instrument_slug}?page={p}'
                print('api_url', api_url)

                data = extract_page_content(api_url)
                if not data or 'results' not in data:
                    continue

                # If no NID and subcategory is 'musical-instruments' → provide a generic text to summarise
                if (not nid or str(nid).strip() == "") and sub_category == 'musical-instruments-of-india':
                    instrument_data = f"This page contains information about the musical instruments"
                else:
                    instrument_data = next(
                        (inst for inst in data['results'] if str(inst.get('nid')) == str(nid)),
                        None
                    )

                if instrument_data:
                    answer = summarise_content(instrument_data, language)
                    return jsonify({'summary': answer}), 200

            return jsonify({"summary": "No NID found in pages 0-4. Try another NID."}), 404

        except Exception as e:
            print("Error in musical instrument handler:", e)
            return jsonify({'summary': 'No NID Found'}), 500


#------------------------------------------------------------------------------------
    def handle_snippets(parsed_url, page, nid, language):
        try:
            # robustly derive the first path segment (subcategory)
            path_parts = [seg for seg in urlparse(parsed_url).path.split('/') if seg]
            sub_category = (path_parts[0].lower() if path_parts else '')

            pages_to_try = [page] if page != "" else list(range(0, 5))
            if page != "" and page not in pages_to_try:
                pages_to_try.extend(range(0, 6))  

            for p in pages_to_try:
                api_url = f'https://icvtesting.nvli.in/rest-v1/snippets?page={p}&&field_state_name_value='
                data = extract_page_content(api_url)

                if not data or 'results' not in data:
                    continue

                # If no NID and subcategory is 'snippets' → set to "test"
                if (not nid or str(nid).strip() == "") and sub_category == 'snippets':
                    subcategory_data = "This page contains information about various snippets"
                else:
                    subcategory_data = next(
                        (category_data for category_data in data['results'] if str(category_data.get('nid')) == str(nid)),
                        None
                    )

                if subcategory_data:
                    answer = summarise_content(subcategory_data, language)
                    return jsonify({'summary': answer}), 200

            return jsonify({'summary': 'No NID found in pages 0-4. Try another NID.'}), 404

        except Exception as e:
            print(e)
            return jsonify({'summary': 'Failed to summarise the page. Try again!'}), 500


#------------------------------------------------------------------------------------

    def handle_stories(parsed_url, page, nid, language):
        try:
            # robustly derive the first path segment (subcategory)
            path_parts = [seg for seg in urlparse(parsed_url).path.split('/') if seg]
            sub_category = (path_parts[0].lower() if path_parts else '')

            pages_to_try = [page] if page != "" else list(range(0, 2))
            if page != "" and page not in pages_to_try:
                pages_to_try.extend(range(0, 3))  

            for p in pages_to_try:
                api_url = f'https://icvtestingold.nvli.in/rest-v1/stories-filter?page={p}&&field_state_name_value='
                print(api_url)
                data = extract_page_content(api_url)

                if not data or 'results' not in data:
                    continue

                if (not nid or str(nid).strip() == "") and sub_category == 'stories':
                    subcategory_data = "This page contains information about various stories available on Indian Cultural Portal."
                else:
                    subcategory_data = next(
                        (category_data for category_data in data['results'] if str(category_data.get('nid')) == str(nid)),
                        None
                    )

                if subcategory_data:
                    answer = summarise_content(subcategory_data, language)
                    return jsonify({'summary': answer}), 200

            return jsonify({'summary': 'No NID found in pages 0-2. Try another NID.'}), 404

        except Exception as e:
            print(e)
            return jsonify({'summary': 'Failed to summarise the page. Try again!'}), 500
    
#------------------------------------------------------------------------------------
    
    # Textiles
    def handle_textiles(parsed_url, page, nid, language):
        
        try:
            base_url = 'https://icvtestingold.nvli.in/rest-v1/textiles-and-fabrics-of-india'
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
                process_type = parsed_url.split('/')[3].lower()
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
                subcategory_data = []
                museum = parsed_url.split('/')[3].split('=')[-1]
                print(museum)
                if museum == 'National-Museum-New-Delhiss' or museum == 'National-Museum-New-Delhi':
                    api_url = 'https://icvtestingold.nvli.in/rest-v1/textiles-and-fabrics-of-india/textiles-museum-collections/national-museum?page={page}&&field_state_name_value='
                elif museum == 'Indian-Museum-Kolkata':
                    api_url = 'https://icvtestingold.nvli.in/rest-v1/textiles-and-fabrics-of-india/textiles-museum-collections/indian-museum?page={page}&&field_state_name_value='
                elif museum == 'Salar-Jung-Museum-Hyderabad':
                    api_url = 'https://icvtestingold.nvli.in/rest-v1/textiles-and-fabrics-of-india/textiles-museum-collections/salarjung-museum?page={page}&&field_state_name_value='
                elif museum == 'Allahabad-Museum-Prayagraj':
                    api_url = 'https://icvtestingold.nvli.in/rest-v1/textiles-and-fabrics-of-india/textiles-museum-collections/ald-msm?page={page}&&field_state_name_value='
                elif museum == 'Victoria-Memorial-Hall-Kolkata':
                    api_url = 'https://icvtestingold.nvli.in/rest-v1/textiles-and-fabrics-of-india/textiles-museum-collections/vmh?page={0}&&field_state_name_value='
                
                data = extract_page_content(api_url)

                for entry in data['results']:
                    subcategory_data.append({'title': entry['title'], 'type': entry['field_dc_type'], 'material': entry['field_dc_format_material'],
                                             'state': entry['field_cdwa_location'] ,'date_issued': entry['field_dc_date_issued'] , 'description': entry['field_dc_description']})
                
                if subcategory_data:
                    answer = summarise_content(subcategory_data, language)               
                    return jsonify({'summary': answer}), 200
                else:
                    return jsonify({'summary': 'No NID Found to fetch data. Try another page'}), 404

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
    # Timeless Trends
    def handle_timeless_trends(parsed_url, page, nid, language):
        try:
            base_url = 'https://icvtestingold.nvli.in/rest-v1/timeless-trends'
            subcategory_type = parsed_url.split('/')[2].lower()
            subcategory_type_1 = parsed_url.split('/')[3].lower()
            print('subcategory_type', subcategory_type)
            
            if subcategory_type == 'accessories':
                api_url = 'https://icvtestingold.nvli.in/rest-v1/timeless-trends/accessories?page=0&&field_state_name_value='
                if 'history' in subcategory_type_1.split('-'):
                    api_url = 'https://icvtestingold.nvli.in/rest-v1/timeless-trends/a-brief-history-section-accessories?page=0&&field_state_name_value='


            if subcategory_type == 'clothing':
                api_url = 'https://icvtestingold.nvli.in/rest-v1/timeless-trends/clothing?page=0&&field_state_name_value='
                if 'history' in subcategory_type_1.split('-'):
                    api_url = 'https://icvtestingold.nvli.in/rest-v1/timeless-trends/a-brief-history-section-clothing?page=0&&field_state_name_value='


            if subcategory_type == 'hairstyles':
                api_url = 'https://icvtestingold.nvli.in/rest-v1/timeless-trends/hairstyles?page=0&&field_state_name_value='
                if 'history' in subcategory_type_1.split('-'):
                    api_url = 'https://icvtestingold.nvli.in/rest-v1/timeless-trends/a-brief-history-section-hairstyle?page=0&&field_state_name_value='


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
                subcategory_data = next((category_data for category_data in data['results'] if str(category_data.get('title').lower().strip()) == subcategory_type_1.replace('-', ' ').lower()), None)

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
            base_url = 'https://icvtestingold.nvli.in/rest-v1/jewellery-of-the-nizams'
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
            base_url = 'https://icvtestingold.nvli.in/rest-v1/unesco'
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
            base_url = 'https://icvtestingold.nvli.in/rest-v1/forts-of-india'
            sub_category = parsed_url.split('/')[2].lower()

            if sub_category == 'discover-forts-of-india':
                api_url = f'{base_url}/discovering-the-forts-of-india?page={page if page != "" else 0}'
            elif sub_category == 'understanding-the-forts':
                api_url = f'{base_url}/understanding-forts?page={page if page != "" else 0}'
            elif sub_category == 'fortsandthefreedomstruggle':
                api_url = 'https://icvtestingold.nvli.in/rest-v1/forts-of-india/forts-and-freedom-struggle?page=0&&field_state_name_value='
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
            base_url = 'https://icvtestingold.nvli.in/rest-v1/ajanta-landing-page'
            api_url = base_url

            if len(parsed_url.split('/')) > 2:
                sub_category = parsed_url.split('/')[2].lower()
                if sub_category == 'paintings':
                    category = 'ajanta-'+sub_category[:-1]
                    api_url = f'https://icvtestingold.nvli.in/rest-v1/{category}?page={page if page != "" else 0}'
                    print(api_url)
                else:
                    category = 'ajanta-'+sub_category
                    api_url = f'https://icvtestingold.nvli.in/rest-v1/{category}?page={page if page != "" else 0}'
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
            base_url = 'https://icvtestingold.nvli.in/rest-v1'
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
                api_url = f'https://icvtestingold.nvli.in/rest-v1/district-repository?page={page}&f%5B0%5D=category_ddr%3ADDR%20Story'
            elif 'Traditions' in parsed_url:
                api_url = f'https://icvtestingold.nvli.in/rest-v1/district-repository?page={page}&f%5B0%5D=category_ddr%3ATraditions%20%26%20Art%20Forms'
            elif 'Personality' in parsed_url:
                api_url = f'https://icvtestingold.nvli.in/rest-v1/district-repository?page={page}&f%5B0%5D=category_ddr%3APersonality'
            elif 'Events' in parsed_url:
                api_url = f'https://icvtestingold.nvli.in/rest-v1/district-repository?page={page}&f%5B0%5D=category_ddr%3AEvents'
            elif 'Treasures' in parsed_url:
                api_url = f'https://icvtestingold.nvli.in/rest-v1/district-repository?page={page}&f%5B0%5D=category_ddr%3AHidden%20Treasures'

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
            base_url = 'https://icvtestingold.nvli.in/rest-v1/retrieved-artefacts-of-india'
            sub_category = parsed_url.split('/')[2].lower()

            print(nid)

            if sub_category in ['reclaimed-relics', 'artefact-chronicles']:
                pages_to_try = [page] if page != "" else list(range(0, 6))
                if page != "" and page not in pages_to_try:
                    pages_to_try.extend(range(0, 5))

                for p in pages_to_try:
                    api_url = f'{base_url}/{sub_category}?page={p}&&field_state_name_value='
                    print(api_url)

                    data = extract_page_content(api_url)
                    if not data or 'results' not in data:
                        continue  
                    print('data', data)

                    # If no NID provided and category is reclaimed-relics → set to "test"
                    if (not nid or str(nid).strip() == "") and sub_category == 'reclaimed-relics':
                        subcategory_data = "Over the years since independence, a multitude of artefacts, numbering in the hundreds, if not thousands, have departed from India's shores, often through unknown covert and clandestine channels. Many lack a clear origin, making it challenging to ascertain their exact place of theft, save for a general notion of their regional provenance. In the past decade, concerted efforts by the Indian government, heritage enthusiasts, and NGOs have led to the repatriation of numerous artefacts to India. This section categorises these artefacts based on their respective sources and according to the materials they are crafted from, although some still bear unknown provenance. Page contains info about reclaimed-relics."
                    else:
                        subcategory_data = next(
                            (category_data for category_data in data['results'] if str(category_data.get('nid')) == str(nid)),
                            None
                        )

                    if subcategory_data:
                        answer = summarise_content(subcategory_data, language)
                        return jsonify({'summary': answer}), 200

                return jsonify({'summary': 'No NID found in pages 0-4. Try another NID.'}), 404

            else:
                return jsonify({'summary': 'The page does not contain information to summarise'}), 404

        except Exception as e:
            print(e)
            return jsonify({'summary': 'Failed to summarise the page. Try again!'}), 500

#-----------------------------------------------------------------------------

    # Freedom Fighters
    def handle_freedom_fighters(parsed_url, page, nid, language):
        try:
            base_url = 'https://icvtestingold.nvli.in/rest-v1/freedom-archive' 
            sub_category = parsed_url.split('/')[2].lower()

            if sub_category == 'unsung-heroes':
                api_url = f'https://icvtestingold.nvli.in/rest-v1/unsung-heroes?page={page if page != "" else 0}&&field_state_name_value='
            if sub_category == 'historic-cities':
                city = parsed_url.split('/')[3].lower().strip()
                if city != 'patna':
                    api_url = f'https://icvtestingold.nvli.in/rest-v1/{city}/Historic_Cities_Freedom_Movement?page=0&&field_state_name_value='
                else:
                    api_url = 'https://icvtestingold.nvli.in/rest-v1/historic-cities/patna/Historic-cities-freedom-movement?page=0&&field_state_name_value='
            if sub_category == 'forts':
                api_url = f'https://icvtestingold.nvli.in/rest-v1/forts-of-india/forts-and-freedom-struggle?page=0&&field_state_name_value='
            if sub_category == 'textile':
                api_url = f'https://icvtestingold.nvli.in/rest-v1/INDIGO-DYE-AND-REVOLT?page=0&&field_state_name_value='
        
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
                api_url = 'https://icvtestingold.nvli.in/rest-v1/cuisine-royal-table?page=0&field_state_name_value='
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
                api_url = 'https://icvtestingold.nvli.in/rest-v1/evolution-cuisine?page=0&&field_state_name_value='
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
            base_url = 'https://icvtestingold.nvli.in/rest-v1/festivals-of-india' 
            category = parsed_url.split('/')[2].lower()
            sub_category = parsed_url.split('/')[3].lower()
            print('cat', category)
            print('sub', sub_category)

            if category == 'pan-indian-festivals':
                api_url = 'https://icvtestingold.nvli.in/rest-v1/festivals-of-India/pan-indian-festivals?page=0&&field_state_name_value='
            if category == 'fairs-and-pilgrimages':
                if sub_category not in ['fairs', 'pilgrimages']:
                    api_url = f'https://icvtestingold.nvli.in/rest-v1/festivals-of-India/fairs-pilgrimages/fairs?page=0&&field_state_name_value='
                if sub_category == 'fairs':
                    api_url = f'https://icvtestingold.nvli.in/rest-v1/festivals-of-India/fairs-pilgrimages/fairs?page=0&&field_state_name_value='
                elif sub_category == 'pilgrimages':
                    api_url = 'https://icvtestingold.nvli.in/rest-v1/festivals-of-India/fairs-pilgrimages/pilgrimage?page=0&&field_state_name_value='
            if category == 'folk-festivals':
                if sub_category not in ['honouring-deities','social-traditions', 'celebrating-nature']:
                    api_url = 'https://icvtestingold.nvli.in/rest-v1/festivals-of-India/folk-festivals/Celebrating-Nature?page=0&&field_state_name_value='
                if sub_category == 'honouring-deities':
                    api_url = 'https://icvtestingold.nvli.in/rest-v1/festivals-of-India/folk-festivals/Honouring-Deities?page=0&&field_state_name_value='
                if sub_category == 'social-traditions':
                    api_url = 'https://icvtestingold.nvli.in/rest-v1/festivals-of-India/folk-festivals/Social-Traditions?page=0&&field_state_name_value='
                if sub_category == 'celebrating-nature':
                    api_url = 'https://icvtestingold.nvli.in/rest-v1/festivals-of-India/folk-festivals/Celebrating-Nature?page=0&&field_state_name_value='
            if category == 'tribal-festivals':
                if sub_category not in ['venerating-ancestors-and-deities', 'worshipping-nature']:
                    api_url = 'https://icvtestingold.nvli.in/rest-v1/festivals-of-India/tribal-festivals/worshipping-nature?page=0&&field_state_name_value='
                if sub_category == 'venerating-ancestors-and-deities':
                    api_url = f'https://icvtestingold.nvli.in/rest-v1/festivals-of-India/tribal-festivals/venerating-ancestors-deities?page=0&&field_state_name_value='
                elif sub_category == 'worshipping-nature':
                    api_url = 'https://icvtestingold.nvli.in/rest-v1/festivals-of-India/tribal-festivals/worshipping-nature?page=0&&field_state_name_value='
           
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
                    api_url = f'https://icvtestingold.nvli.in/rest-v1/north-east-archive/unsung-heroes?page=0&&field_state_name_value=?page={page if page != "" else 0}&&field_state_name_value='
                if sub_category == 'capital-cities-north-east-india':
                    if section == 'shillong':
                        if split == 'history-evolution':
                            api_url = f'https://icvtestingold.nvli.in/rest-v1/north-east-marker-shillong-history?page=0&&field_state_name_value='
                        if split == 'natural-built-heritage':
                            api_url = f'https://icvtestingold.nvli.in/rest-v1/north-east-marker-shillong-natural-and-built?page=0&&field_state_name_value='
                        if split == 'streets-localities':
                            api_url = f'https://icvtestingold.nvli.in/rest-v1/north-east-marker-shillong-street-and-localities?page=0&&field_state_name_value='
                        if split == 'life-in-the-city':
                            api_url = f'https://icvtestingold.nvli.in/rest-v1/north-east-marker-shillong-life-in-the-city?page=0&&field_state_name_value='

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
                        api_url = f'https://icvtestingold.nvli.in/rest-v1/states-of-india/bihar/tidbits-tales-trivia?page=0&&field_state_name_value='
                    if section == 'digital-archives':
                        api_url = 'https://icvtestingold.nvli.in/rest-v1/states-of-india/bihar/digital-archives?page=0&&field_state_name_value='
                    elif section == 'bihar-through-traveller-s-gaze':
                        api_url = 'https://icvtestingold.nvli.in/rest-v1/states-of-india/bihar/bihar-through?page=0&&field_state_name_value='
                        
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
                        api_url = 'https://icvtestingold.nvli.in/rest-v1/art-and-architecture-api-pins?page=0&&field_state_name_value='
                        
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
            api_url = f'https://icvtestingold.nvli.in/rest-v1/iconic-battle-of-india/detail?page=&&field_state_name_value='

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
                api_url = 'https://icvtestingold.nvli.in/rest-v1/legendary-figure/kings-queens?page=0&&field_state_name_value='
            if section == 'social-reformers-and-revolutionaries':
                api_url = f'https://icvtestingold.nvli.in/rest-v1/legendary-figure/social-reformers-revolutionaries?page=0&&field_state_name_value='
            if section == 'sages-philosophers-and-thinkers':
                api_url = 'https://icvtestingold.nvli.in/rest-v1/legendary-figure/sages-philosophers-thinkers?page=0&&field_state_name_value='

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

        nid_ = int(parsed_url.split('/')[-1].split('=')[-1])
        sub_category = parsed_url.split('/')[2].lower().strip()
        tab = ((request.get_json(silent=True) or {}).get('tab') or '').strip().lower()

        print('healing_nid', nid)
        print('step 1 - tab', tab)

        # --- Build API URL ---
        if sub_category == 'pan-indian-traditions':

            if nid_ ==  3000998:
                data = {'history': "ayurveda, one of the world's oldest systems of healing, is believed to be more than 5000 years old. It is rooted in a holistic understanding of health and wellness. The term 'Ayurveda' comes from the Sanskrit words Ayur, meaning life, and Veda, meaning knowledge or science. This ancient system aims to prevent and treat various ailments by harmonizing the body with nature and elements of the universe.The Ayurvedic classics describe eight well developed clinical branches of Ayurveda, these include : Kaya Chikitsa (Medicine), Shalya Tantra (Surgery), Shalakya Tantra (ENT and Ophthalmology), Kaumarbhritya (Paediatrics and Obstetrics), Agad Tantra (Toxicology), Bhut Vidya (Psychiatry), Rasayan  (Rejuvenation therapy and geriatrics), and Vajikaran (Sexology - Including Aphrodisiac for better progeny).The history or origin of Ayurveda is associated with numerous myths and legends. It is believed to have a divine origin, gifted to humanity by the gods to ensure health and longevity. The most prominent figure associated with Ayurveda’s creation is Lord Brahma, the creator god in Hindu mythology. According to various traditions, Brahma imparted this ancient knowledge to sages, who then passed it down to humanity through oral teachings and written texts.As per another version, Dhanvantari is believed to be the god of medicine and healing. He is said to have emerged from the churning of the ocean (Samudra Manthan), carrying a bowl of nectar and a healing staff. Dhanvantari is revered as the divine physician and is often depicted as a central figure in Ayurvedic practice, particularly in the field of surgery. In some versions of the myth, Dhanvantari is also considered a king of Kashi (Varanasi), blessed by Lord Vishnu with the gift of medical knowledge.",
                                'philosophy': "Ayurveda emphasises the harmonious balance between body, mind, spirit, and social well-being. Central to Ayurvedic philosophy is the belief in the interconnectedness of all elements within the universe, which includes space, air, fire, water, and earth. The five great elements or the panchmahabhuta, form the human body and are present in different proportions. Together they form unique physical and psychological characteristics of an individual. This unique constitution is referred to as prakruti or prakriti. Every individual is believed to be formed of two components i.e. Prakriti, which constitutes the body and Purusa, which is the soul.  While Purusa is indivisible and unchangeable, Prakriti evolves and creates from varying combinations. Prakruti is determined by the balance of three fundamental energies or doshas i.e. vata (space and air), pitta (fire and water), and kapha (water and earth). Each dosha embodies distinct qualities that govern bodily functions and personality traits. While the human body comprises all three, one is naturally more prominent than the others. Ayurveda has further classified the human body into Saptadhatus (seven tissues) which are composed of the five elements. It includes Rasa (tissue fluids), Meda (fat and connective tissue), Rakta (blood), Asthi (bones), Majja (marrow), Mamsa (muscle), and Shukra (semen) and three Malas (waste products) of the body, viz. Purisha (faeces), Mutra (urine) and Sweda (sweat). They are interconnected with the doshas.Maintaining balance between the doshas is key to health. Therefore, everyone needs to adhere to daily and seasonal regimens that align with one’s prakruti. Throughout life the overall composition remains the same, but it gets affected by various internal, external and environmental factors. Therefore, everyone is responsible for their choices about diet, exercise, and relationships, which can create physical, emotional, or spiritual imbalances. Ayurveda places great emphasis on the prevention of disease rather than mere treatment. A trained Ayurvedic practitioner conducts a thorough assessment, observing physical traits, behavioural patterns, and pulse characteristics to determine an individual's prakriti and dosha. This personalised approach highlights the philosophy's commitment to understanding the individual’s unique constitution, thus enabling tailored health recommendations.The healing patterns associated with this ancient system reflect a dynamic interplay of internal and external factors. Its principles continue to resonate in contemporary wellness practices, offering a timeless framework for achieving balance and well-being in an increasingly complex world. Over the millennia, Ayurveda has not only persisted as a medical tradition but has also influenced many aspects of cultural and spiritual life in India and beyond. Its holistic approach to health, which seeks to maintain balance between the individual and nature, continues to be relevant in contemporary wellness practices. Ayurveda remains a testament to the ancient wisdom that sought to integrate healing with a broader understanding of the universe, where physical health, mental clarity, and spiritual harmony are seen as inseparable.Over centuries, these texts have been translated, annotated, and studied, influencing not only traditional Ayurvedic practices but also modern healthcare in many parts of the world. Today, they continue to be central to Ayurvedic education and practice, preserving the wisdom of ancient India and adapting it to contemporary needs.",
                                'practitioners': "The ancient wisdom and practices of Ayurveda continue to thrive today through a rich tapestry of institutions, practitioners, and pharmaceutical enterprises. These contemporary expressions are not mere echoes of the past—they actively sustain and reinterpret ancient principles to meet our daily needs. Let's explore some aspects of Ayurveda's enduring legacy across India through this interactive map. Each pin marks a site where this healing tradition has left an indelible mark and expanded the traditional medical paradigm. Together, these sites form a living archive of a healing tradition that has never ceased to innovate and adapt while honouring its deep historical roots. FOllowing are the practitioners: Sage Atreya, Charaka, Sushruta, Agnievesha, Sage Bharadwaj, Sage Atreya, Lord Dhanvantari, Vagbhata, Madhava - Kara, Chakrapani Datta, Bhavamishra",
                                'literature': "Over the centuries Ayurveda has played a crucial role in shaping medical thought and practice. The system's comprehensive approach to health, focusing on the balance of body, mind, and spirit, is documented in a body of literature that dates from approximately 400 BCE to 200 CE. The foundational theories and practices of Ayurveda can be traced to even earlier times, with its roots embedded in ancient Indian philosophical and spiritual traditions.The historical significance of Ayurvedic literature lies not only in its early contributions to the field of medicine but also in its holistic and integrative approach to understanding the human body. As research continues to reveal the depth and relevance of Ayurvedic thought, it is important that these contributions be more widely acknowledged and incorporated into the broader discourse of medical science.These texts are considered the pillars upon which Ayurveda rests. They provide detailed insights into its principles, diagnoses, treatments, and lifestyle practices. Together, these texts codify the vast knowledge of Ayurveda into systematic frameworks for health and healing. They represent the collective wisdom of sages and physicians in ancient India and continue to influence the practice of Ayurvedic medicine to this day. They serve not only as medical guides but also embody the philosophical, cultural, and scientific insights of the past.The foundational texts that form the core of Ayurvedic knowledge are collectively known as the Brihat Trayi (or The Three Great Texts). These three texts are considered the pillars upon which Ayurveda rests, providing detailed insights into its principles, diagnoses, treatments, and lifestyle practices. They are:Charaka Samhita, Sushruta Samhita,Ashtanga Hridaya.The Brihat Trayi represents the pinnacle of Ayurvedic scholarship and are revered by practitioners and scholars of Ayurveda worldwide. Together, these texts form a unified system that addresses all aspects of human health, from the prevention of disease to the treatment of ailments, including both medical and surgical knowledge. Their emphasis on natural remedies, ethical conduct, and holistic health has influenced not only Indian medicine but also global health practices. ",
                                'surgicalequipment': 'The Sushruta Samhita stands as one of historys most groundbreaking medical treatises. It is attributed to the legendary physician Sushruta and provides an unparalleled glimpse into the advanced surgical knowledge of ancient India. It not only details complex procedures, including reconstructive surgery, and cataract removal, but also meticulously describes over hundreds of surgical instruments—many of which resemble modern scalpels, forceps, and catheters.It can be said that the precision of Sushrutas instruments, crafted from materials such as iron and copper, laid the groundwork for the sophisticated surgical tools used today. Some historians and medical experts even consider his work a precursor to modern techniques, as his emphasis on cleanliness, wound care, and surgical ethics aligns closely with contemporary medical practices. Today, as medicine continues to evolve, the principles outlined in this ancient text remain relevant. Let us explore some of the instruments mentioned in Sushruta Samhita. Their knowledge bridges the past and present and also serves as a testament to the scientific spirit of early Indian medicine.The medical instruments were meticulously crafted from metals such as iron, copper, and gold, as well as non-metallic materials sourced from animals and plants. These tools, characterised by their sharp edges and firm grip, were tailored for specific tasks such as incisions (bhedana), excision(chedana), and suturing(seevana) among others.In ancient India, surgical tools were bifurcated into: Yantras (blunt instruments) and Shastras (sharp instruments), each with its own unique purpose. Yantras: Yantras (Blunt Instruments) are shaped like birds and animals, such as lions, tigers, deer, and sparrows, among others. There are 101 yantra instruments, which are further subdivided into six subgroups. Each yantra serves a distinct medical purpose. Shastras: The sharp instruments described in the Sushruta Samhita are referred to as shastras. There are twenty such instruments: the Mandalagra, Karapatra, Vriddhipatra, Nakha, Mudrika, Utpalapatra, Ardhadhara, Suchi, Kushapatra, Aatimukha, Shararimukha, Antarmukha, Trikurchaka, Kutharika, Vrihimukha, Ara, Vetasapatra, Vadisha, Dantashanku, and Eshani.  Each of these instruments had an important role in the practice of ancient Indian surgery. The Sushruta Samhita describes 20 different sharp instruments, each serving a specific surgical function.',
                                'medicalmap': 'The ancient wisdom and practices of Ayurveda continue to thrive today through a rich tapestry of institutions, practitioners, and pharmaceutical enterprises. These contemporary expressions are not mere echoes of the past—they actively sustain and reinterpret ancient principles to meet our daily needs. Enterprise: Zandu, Dabur, Shree Dhootapapeshwar Limited, Baidyanath, K.P. Namboodiris Ayurvedics, Himalaya Wellness. Institution: Government Ayurvedic Medical College Mysore, Arya Vaidya Sala, Kottakkal, Dayanand Ayurvedic College Jalandhar, Lalit Hari State Ayurveda College, Pilibhit, J.B. Roy State Ayurvedic Medical College. Legacy: Ashtavaidya Tradition.'  
                            }
                
                if tab:
                    return jsonify({"tab": tab, "summary": summarise_content(data[tab], language)}), 200
                else:
                    return jsonify({"summary": summarise_content(data, language)}), 200

            if nid_ == 3002086:
                data = {
                    'history': "Unani Tibb is one of the world's oldest healing practices, and it is believed to have roots that span over a thousand years. It represents a complex synthesis of various medical thoughts. The term Unani is derived from the Arabic word Yunan, which refers to the Ionian region i.e. parts of ancient Greece - while Tibb translates to medicine. Unani Tibb represents a complex synthesis of knowledge of healing which was integrated into the cultural fabric of many regions around the world because of its reliable and effective natural remedies. Unani Tibb's rich and long history makes it a deeply researched and tested healing practice which promotes well-being and harmony in the body. It is believed to have its roots in ancient Greek medical knowledge. Its theoretical framework is based on the teachings of Greek physicians & scholars such as Hippocrates (460-370 BC) and Galen (129-200 CE). Between the 8th and 13th centuries, Greek medical knowledge was transmitted and formalised in the Arab and Persian world. Under the patronage of Abbasid caliphs, it was further refined through observation, translation of existing Greek literature and experimentation by scholars. This led to the foundation of Bayt al-Hikma (The House of Wisdom), which became a major intellectual and academic center in Baghdad during the Abbasid Caliphate. It functioned as both a library, a center for scholarly activities and translation from Greek, Persian, and other languages into Arabic. Due to the connection between the two regions, this healing tradition is also known as Greco- Arab medicine. Physicians like Ibn Sina, known in the West as Avicenna, contributed to this tradition through his most celebrated work called Al-Qānūn fil-ṭibb or Canon of Medicine. Another eminent scholar, Al-Razi, known in the West as Rhazes, wrote Al-Hawi fi al-Tibb also known as ‘The Comprehensive book’, and many others were written which assisted in the dissemination of Unani Tibb. These Arab scholars helped to synthesize the Hippocratic and Galenic knowledge within the epistemological frameworks of Arabic philosophy.In the 8th century, Unani Tibb was introduced to India with the arrival of Arab scholars and hakeems (Unani practitioners). The tradition gained its strong foundation under the patronage of the Delhi Sultanate and the Bahmani Kingdom in Southern India. One of the first works of Greco-Arab medicine in India was a 13th-century Persian translation by Abu Bakir bin Ali bin Usman of Alberuni’s Kitab al-Saydana fi'l-Tibb (The Book of Pharmacy in Medicine). This translation was patronised by Sultan Iltutmish of the Delhi Sultanate (1211-1236).The establishment of the Mughal empire during the 16th century marked another major phase in the development of Unani Medicine in India. Rulers like Babur and Humayun brought tabibs (physicians) from their native land. Medical learning, practice and literature flourished during the reign of subsequent rulers both in North and South India. During this time, Unani Tibb was widely practised in India alongside other traditional systems of medicine such as Ayurveda and Siddha. The medical pluralism of the time allowed for a dynamic exchange of ideas and methods.In the 18th century, with the decline of the Mughal Empire and the establishment of British colonial rule, there was a shift from the traditional medical system to Western medicine, which was promoted by the British colonists. However, with the disintegration of the Mughal empire, princely states like Rampur, Bhopal, Hyderabad and Lucknow became important centres for learning and practice of Unani medical tradition. Notable families like Sharif and Majeedi from Delhi and Azizi from Lucknow helped to preserve and promote it throughout the 19th and 20th centuries. In present times, Unani Tibb continues to thrive under the ministry of AYUSH (Ayurveda, Yoga & Naturopathy, Unani, Siddha, and Homoeopathy).",
                    'literature': "Unani Tibb, rooted in the classical philosophies of ancient Greece and Rome, was later enriched by Persian, Arab, and South Asian scholars. It consists of a sophisticated body of literature. The composition of Unani texts over time reflects the cumulative efforts of generations of physicians, philosophers, and translators who preserved, critiqued, and expanded upon age-old medical wisdom. The intellectual foundations of Unani Tibb were laid in the classical era by Greek physicians, particularly Hippocrates (c. 460–370 BCE), who is often hailed as the 'Father of Medicine.' His writings, collectively known as the Hippocratic Corpus, introduced critical ideas such as the humoral theory of health and disease, the importance of clinical observation, and ethical conduct in medicine. This work emphasised the balance of the four humors—blood, phlegm, yellow bile, and black bile, a concept that remained central to Unani philosophy. Later, Galen of Pergamon (129–c. 200 CE) systematised and expanded Hippocratic principles by producing a vast medical corpus on anatomy, physiology, pathology, pharmacology, and philosophy. His texts, originally written in Greek and were later translated into Latin and Arabic. These works became foundational references in both Islamic and European medical education for centuries.The rise of Islamic civilization from the 7th century onward marked a transformative phase in the development of Unani medicine. In centers of learning such as Baghdad, Neyshabur, Cairo, a major translation movement flourished under the patronage of the Abbasid caliphs. Greek medical texts were translated into Arabic by scholars such as Hunayn ibn Ishaq and Thabit ibn Qurra, ensuring their survival and integration into the region’s intellectual tradition. These Arab and Persian scholars did not merely translate the classical literature, but they critically engaged with it, producing original medical works which refined earlier theories through empirical observation and philosophical reasoning. Among the most influential figures of this era was Abu Bakr Muhammad ibn Zakariya al-Razi, (865–925 CE) also referred to as Rhazes in the West, who compiled a monumental medical encyclopedia, Al-Hawi (The Comprehensive Book).  His collection of work primarily focused on clinical cases, therapies, and pharmacological data.Another pioneer in this domain was Abu Ali al-Husayn ibn Sina (980–1037 CE), known in the West as Avicenna. His magnum opus, Al-Qanun fi al-Tibb (The Canon of Medicine), synthesized Galenic, Hippocratic, and Arabic medical knowledge into a collective and systematic medical philosophy. The Canon was organised into five books: Book I-General principles, Book II-Materia medica; Book III-Diseases of the individual organs; Book IV-General diseases; Book V-Formula for remedies. It remained the authoritative text in the Arab world until the early modern period.The literary output of Unani scholars during the Abbasid Caliphate period (8th -13th century) was vast and varied in form. Medical texts were composed not only as encyclopedias but also as abridgements, enabling easier transmission of complex ideas. Commentaries were frequently written on classical texts, particularly those of Hippocrates and Galen, to clarify and expand upon their meanings. A unique feature of the Unani tradition was the use of poetry, where complex medical knowledge was expressed in verse. Scholars also wrote monographs— focused treatises on specific diseases, organs, or therapies—and therapeutic manuals that detailed diagnostic techniques, clinical signs, and appropriate regimens of treatment.Pharmacological literature, another key genre in Unani Tibb, developed extensively by this time. Works like Kitab al-Jami fi al-Adwiya al-Mufrada (The Book of Simple Drugs) by Ibn al-Baytar catalogued hundreds of natural substances, including herbs, minerals, and animal products, along with their therapeutic properties and uses. These pharmacopoeias, often organised alphabetically or thematically, provided the foundation for the Unani system of compound drug preparation and were central to the practice of hakims (Unani physicians). Texts on dietetics and regimen were also widely produced, emphasising the importance of food, environment, and lifestyle in maintaining health and preventing disease.Manuscripts of these texts were often exquisitely produced, written in elegant scripts, bound in decorated leather covers illustrated with botanical drawings or anatomical diagrams. These rare works were well preserved in libraries, royal courts, and madrasas (Islamic educational institutions).The enduring appeal of Unani medical literature lies in its humanistic approach to medicine. It integrates philosophy, natural science, and empirical observation, while emphasising balance, temperament (mizaj), and individualised care. Today, these texts remain not only as historical artifacts but as living documents in traditional systems of healing practiced across South Asia and the Middle East. Several important repositories in Indian institutions preserve the rich legacy of Unani medical manuscripts. The Rampur Raza Library in Rampur, Uttar Pradesh, holds one of the largest collections of Persian medical texts in the country, including rare and early works. The Khuda Bakhsh Oriental Public Library in Patna, Bihar, is equally renowned for its extensive holdings in Arabic and Persian Unani literature, making it a vital resource for historical research. In Delhi, the Jamia Hamdard Library houses numerous rare, printed editions and manuscript texts that continue to support Unani medical education and scholarship.Meanwhile, the Darul Shifa College (16th century) and  Nizamia Tibbi College in Hyderabad, are believed to be the oldest Unani medical institutions in South India. They serve as an important repository for works originating from the Deccan region and reflect the historical depth of Unani practice in southern India. Manuscripts from the Mughal era are also available in State Central Library, Hyderabad. These institutions not only preserve the literary legacy of Unani medicine but also stand as enduring symbols of its evolution.",
                    'practitioners': "The rich legacy of Unani medicine has been carefully preserved and transmitted through generations by dedicated practitioners. These esteemed healers not only compiled their vast knowledge into influential texts but also guided and trained others in medicine or the art of healing. Few of them are: Ibn Sina (Avicenna), Abu Bakr al-Razi,Hakim Mohammad Sharif Khan, Hakim Ajmal Khan, Hakim Hashim Alavi Shirazi, Hakim Akbar Arzani, Hakim Ali Gilani, Muhammad ibn Yusuf al Harawi, Hakim Abdul Aziz, Hakim Abdul Hameed, Hakim Mohammad Kabiruddin, Hakim Syed Khaleefathullah",
                    'medicalmap': "Institution: Dar- ul- shifa, Hyderabad, Takmeel-ut-Tibb, Lucknow, Ajmal Khan Tibbiya College - Aligarh Muslim University (AMU) Aligarh, Ayurvedic and Unani Tibbia College Delhi, Government Nizamia Tibbia College and Hospital HyderabadState Unani Medical College and Hospital Prayagraj, Dar- ul- shifa Hyderabad, Takmeel-ut-Tibb Lucknow. Enterprises: Hamdard Laboratories Delhi, Khandan-e-Sharifi (Sharifi Family of Delhi), Baqai Dawakhana Pvt. Ltd. Delhi, Tayyebi Dawakhana Indore, Dar-ul-Shifa Mandu",
                    'philosophy': "Unani Tibb is a Greco-Arabic system of medicine rooted in the synthesis of ancient Greek humoral theory, especially the works of Hippocrates and Galen, and later enriched by Islamic philosophers and physicians like Ibn Sina (Avicenna), and Al-Razi (Rhazes). It is based on a holistic understanding of cosmology, physiology, and the relationship between humans and nature. This system of medicine is fundamentally based on the concept of the four humors—phlegm, blood, yellow bile, and black bile. —which are believed to influence an individual's temperament and health. These humors circulate throughout the body and influence not only physical health but also personality traits and emotional disposition. The word humor is derived from the Greek word chymos, which translates to juice or bodily fluid. In this medical tradition, health and disease are explained through seven governing principles, known as Umoor-e-Tabiyah. These natural and physiological factors regulate the human body, sustain well-being, and, when disturbed, give rise to illness. Click on the icons below to explore each principle in detail: Blood => Arkan =>In this medical tradition, health and disease are explained through seven governing principles, known as Umoor-e-Tabiyah. These natural and physiological factors regulate the human body, sustain well-being, and, when disturbed, give rise to illness. Click on the icons below to explore each principle in detail: The four foundational elements—Earth, Water, Air, Fire-forming all matter including the human body, Temperament => Mizaj => The unique combination of qualities in a person formed from humors, Humors => Akhlat => The four bodily fluids that regulate health and illness, Organs => Aza => Structural parts of the body responsible for specific functions, Vita Spirit => Arwah => The life force or pneuma enabling physiological activities, Faculties => Quwa => Powers responsible for different bodily actions (e.g., natural, vital, psychic), Functions => Af'al => Actions or functions performed by organs (e.g., digestion, circulation). Unani’s humoral philosophy views nature and mankind as ideologically coexisting in a balanced manner. Each individual has a unique temperament resulting from the dominant qualities of their humors. Nature too affects the human constitution and the four humors.A healthy body is believed to be a state of equilibrium, extreme excess or deficiency in the humors creates an imbalance which leads to illness or sickness.Diagnosis in Unani medicine involves a holistic evaluation of the patient's constitution, symptoms, and lifestyle, using the pulse examination (nabz), urine analysis (baul), stool inspection (baraz), observation of physical signs: tongue, complexion, eyes, nails, voice and assessment of temperament. Once the imbalance is identified, it is resolved based on both preventive and curative approaches.  In the traditional system, treatments are individualistic i.e. based on the nature of the disease, the patient’s temperament, and the humoral imbalance. The five principal treatment modalities are:Regimental Therap Arabic / Persian Term: Ilaj-bit-Tadbeer. Description: Lifestyle modifications and physical treatments to improve health and balance humors.Common Techniques: Exercise (Riyazat), Massage (Dalak), Cupping (Hijama), Venesection (Fasd), Steam Bath (Hammam),Dietotherapy.Arabic / Persian Term: Ilaj-bil-Ghiza Description: Personalized dietary plans based on disease and temperament.Common Techniques: Use of therapeutic foods from plant and animal sources.,Pharmacotherapy Arabic / Persian Term: Ilaj-bid-Dawa. Description: Use of natural drugs to restore humoral balance. Common Techniques: Herbal, mineral, and animal-derived formulations.,Concoctive and Purgative Therapy Arabic / Persian Term: Munzij wa Mushil. Description: Two-step process: Munzij drugs soften and mature morbid matter; Mushil drugs expel it. Common Techniques: Used for chronic conditions or when waste accumulates in the body.,Surgery Arabic / Persian Term: Ilaj-bil-Yad Description: Applied in emergencies or when all other treatments fail.Common Techniques: Incisions, excisions, drainage, and bone setting."
                }

                if tab:
                    return jsonify({"tab": tab, "summary": summarise_content(data[tab], language)}), 200
                else:
                    return jsonify({"summary": summarise_content(data, language)}), 200
                
        elif sub_category == 'unconventional-traditions':
            api_url = (
                "https://icvtestingold.nvli.in/rest-v1/healing-through-the-ages/"
                f"unconventional-traditions?page={page if page else 0}"
                "&field_state_name_value="
            )

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
        
        else:
            return jsonify({"summary": "Unknown subcategory"}), 400

            
            # print('page content extracted')

            # sub_data = next((cd for cd in data['results'] if str(cd.get('nid')) == str(nid)), None)
            # if not sub_data:
            #     return jsonify({'summary': 'No NID found to fetch data. Try another page'}), 404

            # print('sub data extracted')

            # --- Map tabs to fields ---
            # TAB_FIELDS = {
            #     "history": "body",
            #     "philosophy": "field_philosophy",
            #     "practitioners": "field_practitioners",
            #     "literature": "field_literature",
            #     "surgical_equipment": "field_surgical_equipment",
            #     'medical_map'
            # }

            # --- Helper: strip HTML and clip ---
        #     def strip_and_clip(text, max_words=150):
        #         if not text:
        #             return ""
        #         text = re.sub(r"<[^>]+>", " ", text)
        #         text = re.sub(r"\s+", " ", text).strip()
        #         words = text.split()
        #         return " ".join(words[:max_words])
            
        #     print('strip and clip done')

        #     tabs = {k: strip_and_clip(sub_data.get(v)) for k, v in TAB_FIELDS.items() if sub_data.get(v)}

        #     print('tabs done')

        #     if tab:
        #         if tab not in tabs:
        #             return jsonify({"summary": f"No content for tab '{tab}'"}), 404
        #         return jsonify({"tab": tab, "summary": summarise_content(tabs[tab], language)}), 200

        #     print('everything done')

        #     return jsonify({"summary": summarise_content(tabs, language)}), 200

        # except Exception as e:
        #     print(e)
        #     return jsonify({'summary': 'Failed to process the page. Try again!'}), 500

#-----------------------------------------------------------------------------

    # classical dances
    def handle_classical_dances(parsed_url, page, nid, language):
        try:
            api_url = f'https://icvtestingold.nvli.in/rest-v1/classical-dances-details?page={page if page != "" else 0}&field_state_name_value='
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
            parts = parsed_url.strip('/').split('/')

            city = parts[-1].lower().strip()
            
            if 'historic-cities-freedom-movement' in city:
                print('here')
                city = city.split('-')[-1]
                print(city == 'ahmedabad')
                if city == 'ahmedabad':
                    api_url = 'https://icvtestingold.nvli.in/rest-v1/ahmedabad/Historic_Cities_Freedom_Movement?page=0&&field_state_name_value='

                if city == 'varanasi':
                    api_url = 'https://icvtestingold.nvli.in/rest-v1/varanasi/Historic_Cities_Freedom_Movement?page=0&&field_state_name_value='
                
                if city == 'lucknow':
                    api_url = 'https://icvtestingold.nvli.in/rest-v1/lucknow/Historic_Cities_Freedom_Movement?page=0&&field_state_name_value='

                if city == 'patna':
                    api_url = 'https://icvtestingold.nvli.in/rest-v1/historic-cities/patna/Historic-cities-freedom-movement?page=0&&field_state_name_value='

                print(api_url)

                data = extract_page_content(api_url)

                if not data or 'results' not in data:
                    return jsonify({"summary": "No data found"}), 404
                
                else:
                    answer = summarise_content(data['results'], language)
                    return jsonify({'summary': answer}), 200

            # Construct the API URL
            api_url = 'https://icvtestingold.nvli.in/rest-v1/historic-cities?page=0&&field_state_name_value='
            print('api_url:', api_url)
            
            # Fetch API data
            data = extract_page_content(api_url)

            # Validate data
            if not data or 'results' not in data:
                return jsonify({"summary": "No data found"}), 404

            # Match subcategory by title
            subcategory_data = next(
                (category_data for category_data in data['results']
                if category_data.get('title', '').lower().strip() == city),
                None
            )

            print(subcategory_data)

            if subcategory_data:
                answer = summarise_content(subcategory_data, language)
                return jsonify({'summary': answer}), 200
            else:
                return jsonify({'summary': 'No matching city found. Try another page'}), 404

        except Exception as e:
            print('Error in handle_historical_cities:', e)
            return jsonify({'summary': 'Failed to summarise the page. Try again!'}), 500
#-----------------------------------------------------------------------------

    def handle_historic_cities(parsed_url, page, nid, language):
        print('func called')
        try:
            api_url = 'https://icvtestingold.nvli.in/rest-v1/historic-cities/delhi/Historic_Cities_Freedom_Movement?page=0&&field_state_name_value='
        
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
                sub_sub_category = parsed_url.split('/')[3].lower().strip()
                type = sub_sub_category.replace('-', '_')
                api_url = f'https://icvtestingold.nvli.in/rest-v1/folktales-of-india/fables?Fables_type={type}'
            if sub_category == 'fairytales':
                api_url = 'https://icvtestingold.nvli.in/rest-v1/fairytales-landing-main?page=0&&field_state_name_value='
            if sub_category == 'legends':
                api_url = 'https://icvtestingold.nvli.in/rest-v1/folktales-of-india/legends?page=0&&field_state_name_'

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
                api_url = 'https://icvtestingold.nvli.in/rest-v1/legendary-figure/kings-queens?page=0&&field_state_name_value='
            
            if sub_category == 'sages-philosophers-and-thinkers':
                api_url = 'https://icvtestingold.nvli.in/rest-v1/legendary-figure/social-reformers-revolutionaries?page=0&&field_state_name_value='
            
            if sub_category == 'social-reformers-and-revolutionaries':
                api_url = 'https://icvtestingold.nvli.in/rest-v1/legendary-figure/sages-philosophers-thinkers?page=0&&field_state_name_value='

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
