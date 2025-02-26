from langgraph.prebuilt import create_react_agent
from llms import search_llm
from langchain_core.messages import SystemMessage, HumanMessage
import requests
from typing import List
from datetime import datetime, timedelta
from state import State
import time
import random
import os

from utils import cosine_similarity, get_openai_embedding, scrape_with_webbase


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

# Web search function used to pull top news links
def web_search(query: str, api_key, count=5, country="US", lang="en") -> List[dict]:
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
            elif (title_similarity > 0.78 or desc_similarity > 0.78) and (selected_date > one_week_ago): 
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

# Web node with enforced 4 valid articles
def web_node(state: State):
    """
    Web search node responsible for gathering exactly 4 links and summaries based on search queries.
    """
    print('-----Running Web Node-----')
    # Pulling state locally
    local_state = state.model_copy()

    # Looping through the topics and performing web search query 
    for i in range(len(local_state.web_queries)):
        print('Performing web search for: ', local_state.desired_subjects[i])
        valid_result: bool = False
        while not valid_result:
            # Querying news search
            chosen_article: dict = None
            time.sleep(1)
            article_urls = web_search(local_state.web_queries[i], api_key=os.getenv("BRAVE_API_KEY"))

            # Checking result of article_url and retrying if failed
            if article_urls is None:
                print('No article found, retrying.')
                time.sleep(1)
                article_urls = web_search(local_state.web_queries[i], api_key=os.getenv("BRAVE_API_KEY"), count=10)
                if article_urls is None:
                    print('Failed to find article for topic. Moving onto next topic.')
                    break
                else:
                    chosen_article = random.choice(article_urls)

            # Checking result of article_url and choosing a random article
            else:
                chosen_article = random.choice(article_urls)

            # Scraping page to gather web content
            print('Article found. Scraping web page.')
            article_content = scrape_with_webbase(chosen_article['url'])

            if article_content and chosen_article:
                # Generating article summary using web agent
                print('Generating summary.')
                local_state.messages = [(HumanMessage(content=f"The article description: {chosen_article['desc']} | The article raw text {article_content}"))]
                result = web_agent.invoke(local_state)
                new_message = result['messages'][-1].content
                # Appending results
                if len(new_message) > 0:
                    local_state.web_links.append(chosen_article['url'])
                    local_state.web_summaries.append(new_message)
                    valid_result = True
            else:
                print('Failed to generate summary, moving on to next topic.')
                break

    # Updating next in local state
    if (len(local_state.web_links) > 0 and len(local_state.web_summaries) > 0 and len(local_state.web_links) == len(local_state.web_summaries)):
        local_state.next = "video"
    else:
        local_state.next = "__end__"

    # Returning updated state and next node
    return local_state