from typing import List
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from temp_obj import topics
import random
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel
from langchain_core.messages import BaseMessage
import re
import ast
import os
import requests
from dotenv import load_dotenv
from langchain.document_loaders import WebBaseLoader
from datetime import datetime, timedelta
import openai
import numpy as np


# Loading env variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


# Loading llms for agent
llm = ChatOpenAI(model = "gpt-4o", temperature = 1.0)
search_llm = ChatOpenAI(model = "gpt-4o", temperature = 0.0)

# preparation_agent, "web_agent", "youtube_agent", "writing_agent", "proof_reading_agent"

# Defining the state dict and the router
class State(BaseModel):
    # For Preparation
    messages: List[BaseMessage] = []
    desired_subjects: List[str] = []
    web_queries: List[str] = []
    youtube_queries: List[str] = []
    # For Search
    web_links: List[str] = []
    youtube_links: List[str] = []
    web_summaries: List[str] = []
    youtube_summaries: List[str] = []
    # For All
    next: str = None


# Defining the preparation agent
preparation_agent = create_react_agent(
    model=llm, tools=[], state_modifier=SystemMessage(content="""
    Your function is to suggest new topics of interest until you have 4 topics. Given a list of desired_subjects. 
    you should determine how many additional topics that you need to suggest. Once you determine this number,
    suggest topics that are similar to the ones initially provided. Also, if any of the topics would be deemed illegal,
    replace them with another suggested topic.

    For example, if ['NBA', 'Murder'] is provided, you could suggest topics like 'NFL', 'NHL', 'WNBA' and respond with
    ['NBA', 'NFL', 'NHL', 'WNBA']                                                                                                    

    Your response should start with: TOPICS: [topic1, topic2, topic3, topic4].
    """)
)


# Defining preparation node
def preparation_node(state: State):
    """
    Preparation node responsible for gathering topics, suggesting a topic, determining if topics are appropriate, and return search queries.
    """
    # Pulling topics from database (for now just pulling from a dict)
    topic_length = len(topics)
    desired_subjects: List[str] = []
    if topic_length >= 3:
        desired_subjects = random.sample(topics, 3)
    else:
        desired_subjects = topics

    # Updating local state
    updated_state = state.model_copy(update={"desired_subjects": desired_subjects})
    updated_state.messages.append(HumanMessage(content=f"desired_subjects: {desired_subjects}"))

    print('Original_topics: ', desired_subjects)

    # Generating suggested topics and replacing innapropriate topics
    result = preparation_agent.invoke(updated_state)
    new_message = result['messages'][-1].content

    # Inserting new topics to local state
    match = re.search(r"TOPICS:\s*(\[[^\]]+\])", new_message)
    if match:
        topics_str = match.group(1)  # Extract the list as a string
        updated_state.desired_subjects = ast.literal_eval(topics_str)  # Convert string to a Python list

    # Generating search queries
    for topic in updated_state.desired_subjects:
        web_query = f'{topic} news'
        youtube_query = f'{topic} news'
        updated_state.web_queries.append(web_query)
        updated_state.youtube_queries.append(youtube_query)

    # Updating message and next in local state
    updated_state.messages.append(SystemMessage(content=f"Suggested topics: {result}"))
    if (len(updated_state.desired_subjects) == 4 and len(updated_state.web_queries) == 4 and len(updated_state.youtube_queries) == 4):
        updated_state.next = "web"
    else:
        updated_state.next = "__end__"

    # Returning updated state and next node
    print('Topics Chosen: ', updated_state.desired_subjects)
    return updated_state


# Web agent
web_agent = create_react_agent(
    model=search_llm,
    tools=[],
    state_modifier=SystemMessage(content="""
    Your function is to summarize articles from the web. You will be given a short 
    description of an article and the raw parsed text from the web page. 
                                 
    Use both of these inputs to generate a two paragraph summary of the article. 
    Make sure you highlight the key points. Return your summary as a string. Include 
    as much detail as possible.
    """)
)


def get_openai_embedding(text: str):
    """Generates an embedding using OpenAI's `text-embedding-ada-002` model."""
    response = openai.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return np.array(response.data[0].embedding)

def cosine_similarity(vec1, vec2):
    """Computes cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


# Web search function used to pull top news links
def web_search(query: str, api_key, count=5, country="us", lang="en") -> List[dict]:
    """Takes in a query and returns a url of a new article"""
    url = "https://api.search.brave.com/res/v1/news/search"
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": api_key
    }
    params = {
        "q": query,
        "count": count,
        "country": country,
        "search_lang": lang,
        "spellcheck": 1
    }

    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        result = response.json()
        relevant_stories: List[dict] = []
        for story in result['results']:
            # Gathering story values
            selected_story = story['url']
            selected_title = story['title']
            selected_description = story['description']
            selected_date = datetime.strptime(story['page_age'], "%Y-%m-%dT%H:%M:%S")
            one_week_ago = datetime.now() - timedelta(days=7)
            stripped_query = query.replace(" news", "")

            # Generating embeddings
            query_embedding = get_openai_embedding(stripped_query)
            title_embedding = get_openai_embedding(selected_title)
            desc_embedding = get_openai_embedding(selected_description)
            title_similarity = cosine_similarity(query_embedding, title_embedding)
            desc_similarity = cosine_similarity(query_embedding, desc_embedding)

            # Checking relevency
            if (stripped_query in selected_title or stripped_query in selected_description) and (selected_date > one_week_ago):
                relevant_stories.append({
                    "url": selected_story,
                    "desc": selected_description
                })
            elif (title_similarity > 0.7 or desc_similarity > 0.7) and (selected_date > one_week_ago): 
                relevant_stories.append({
                    "url": selected_story,
                    "desc": selected_description
                })
        if len(relevant_stories) > 0:
            return relevant_stories
        else:
            return None
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None


# Web scraping function used to gather content from given url
def scrape_with_webbase(url: str) -> str:
    """
    Used to scrape content from a URL. Given a url it returns a string representation of the page content.
    """
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()
        return docs[0].page_content if docs else "No readable content found."
    except Exception as e:
        return f"Failed to fetch the webpage: {e}"
    

# Web node with enforced 4 valid articles
def web_node(state: State):
    """
    Web search node responsible for gathering exactly 4 links and summaries based on search queries.
    """
    # Pulling state locally
    local_state = state.model_copy()

    # Looping through the topics and performing web search query 
    for i in range(len(local_state.web_queries)):
        valid_result: bool = False
        while not valid_result:
            # Querying news search
            chosen_article: dict = None
            article_urls = web_search(local_state.web_queries[i], api_key=os.getenv("BRAVE_API_KEY"))

            # Checking result of article_url and retrying if failed
            if article_urls is None:
                article_urls = web_search(local_state.web_queries[i], api_key=os.getenv("BRAVE_API_KEY"), count=10)
                if article_urls is None:
                    break

            # Checking result of article_url and choosing a random article
            else:
                chosen_article = random.choice(article_urls)

            # Scraping page to gather web content
            article_content = scrape_with_webbase(chosen_article['url'])

            if article_content and chosen_article:
                # Generating article summary using web agent
                local_state.messages = [(HumanMessage(content=f"The article description: {chosen_article['desc']} | The article raw text {article_content}"))]
                result = web_agent.invoke(local_state)
                new_message = result['messages'][-1].content
                # Appending results
                if len(new_message) > 0:
                    local_state.web_links.append(chosen_article['url'])
                    local_state.web_summaries.append(new_message)
                    valid_result = True
            else:
                article_urls = web_search(local_state.web_queries[i], api_key=os.getenv("BRAVE_API_KEY"), count=10)
                if article_urls is None:
                    break

    # Updating next in local state
    if (len(local_state.web_links) > 0 and len(local_state.web_summaries) > 0 and len(local_state.web_links) == len(local_state.web_summaries)):
        local_state.next = "video"
    else:
        local_state.next = "__end__"

    # Returning updated state and next node
    return local_state


# Defining the youtube agent


# Video search function used to pull top youtube links
def video_search(query: str, api_key, count=5, country="us", lang="en", freshness="pw") -> List[dict]:
    """Takes in a query and returns a url of a youtube video"""
    url = "https://api.search.brave.com/res/v1/videos/search"
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": api_key
    }
    params = {
        "q": query,
        "count": count,
        "country": country,
        "search_lang": lang,
        "spellcheck": 1,
        "freshness": freshness
    }

    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        result = response.json()
        relevant_videos: List[dict] = []
        for video in result['results']:
            # Gathering video values
            selected_video: str = video['url']
            selected_title: str = video['title']
            selected_description : str = video['description']
            selected_date = datetime.strptime(video['page_age'], "%Y-%m-%dT%H:%M:%S")
            one_week_ago = datetime.now() - timedelta(days=7)
            stripped_query = query.replace(" news", "")

            # Generating embeddings
            query_embedding = get_openai_embedding(stripped_query)
            title_embedding = get_openai_embedding(selected_title)
            desc_embedding = get_openai_embedding(selected_description)
            title_similarity = cosine_similarity(query_embedding, title_embedding)
            desc_similarity = cosine_similarity(query_embedding, desc_embedding)

            # Checking relevency
            if (stripped_query in selected_title or stripped_query in selected_description) and (selected_date > one_week_ago) and ('https://www.youtube.com' in selected_video):
                relevant_videos.append({
                    "url": selected_video,
                    "desc": selected_description
                })
            elif (title_similarity > 0.7 or desc_similarity > 0.7) and (selected_date > one_week_ago) and ('https://www.youtube.com' in selected_video): 
                relevant_videos.append({
                    "url": selected_video,
                    "desc": selected_description
                })
        if len(relevant_videos) > 0:
            return relevant_videos
        else:
            return None
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None


# Defining youtube node
def youtube_node(state: State):
    """
    YouTube search node responsible for gathering exactly 4 links and summaries for youtube videos.
    """
    # Pulling state locally
    local_state = state.model_copy()
    
    # Looping through the topics and performing video search query 
    for i in range(len(local_state.youtube_queries)):
        valid_result: bool = False
        while not valid_result:
            # Querying video search
            chosen_video: dict = None
            video_urls = video_search(local_state.youtube_queries[i], api_key=os.getenv("BRAVE_API_KEY"))

            # Checking result of video_url and retrying if failed
            if video_urls is None:
                video_urls = video_search(local_state.web_queries[i], api_key=os.getenv("BRAVE_API_KEY"), count=10)
                if video_urls is None:
                    break

            # Checking result of video_url and choosing a random video
            else:
                chosen_video = random.choice(video_urls)

            breakpoint()
            # TODO - Add retry for API rate limiting or try to sleep between each loop to avoid
            # TODO - Figure out how to get video summaries -> 


# Defining the conditional edge router
def router(state: State):
    current_state = state.model_copy()
    if current_state.next == 'web':
        return 'web_node'
    elif current_state.next == 'video':
        return 'youtube_node'
    elif current_state.next == '__end__':
        return '__end__'
    else:
        return '__end__'


builder = StateGraph(State)
builder.add_edge(START, "preparation_node")
builder.add_node("preparation_node", preparation_node)
builder.add_conditional_edges('preparation_node', router)
builder.add_node('web_node', web_node)
builder.add_conditional_edges('web_node', router)
builder.add_node('youtube_node', youtube_node)
builder.add_edge('youtube_node', END)
graph = builder.compile()
    



async def invoke_agent():

    initial_state = State()

    result = graph.invoke(initial_state)

    breakpoint()

    return result
