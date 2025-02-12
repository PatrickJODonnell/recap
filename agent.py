from typing import Literal, List
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from temp_obj import topics
import random
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel
from langchain_core.messages import BaseMessage
import re
import ast
from langchain_community.tools.tavily_search import TavilySearchResults
import json

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
        web_query = f'Find a recent article about {topic} related news within the last 3 days.'
        youtube_query = f'Find a recent video about {topic} related news within the last 3 days.'
        updated_state.web_queries.append(web_query)
        updated_state.youtube_queries.append(youtube_query)

    # Updating message and next in local state
    updated_state.messages.append(SystemMessage(content=f"Suggested topics: {result}"))
    if (len(updated_state.desired_subjects) == 4 and len(updated_state.web_queries) == 4 and len(updated_state.youtube_queries) == 4):
        updated_state.next = "search"
    else:
        updated_state.next = "__end__"

    # Returning updated state and next node
    return updated_state

# Function to validate an article link
def validate_link(link: string) -> bool:
    """Checks the link to see if it fits any generic page criteria"""
     # Patterns indicating an article (dates, detailed slugs)
    article_patterns = [
        r'/\d{4}/\d{1,2}/\d{1,2}/',  # Matches YYYY/MM/DD format in URL
        r'/\d{4}/\d{1,2}/',          # Matches YYYY/MM format in URL
        r'/\d{4}/'                   # Sometimes just a year in the path is enough
    ]

    # Patterns indicating a general page (news indexes, team pages, etc.)
    general_page_patterns = [
        r'/team/', r'/topics/', r'/news/', r'/latest/', r'/category/', r'/section/',
        r'/overview/', r'/index.html', r'/home/', r'/archives/', r'/tag/', r'/stories/',
        r'/updates/', r'/all-news/', r'/sports/', r'/league/', r'/standings/', r'/schedule/',
        r'/all/'
    ]

    # Check if it matches an article pattern
    is_article = any(re.search(pattern, link) for pattern in article_patterns)
    
    # Check if it matches a general page pattern
    is_general_page = any(re.search(pattern, link) for pattern in general_page_patterns)

    # Final classification
    if is_article and not is_general_page:
        return True
    elif is_article is is_general_page:
        return True


# Function to parse and validate the agent's response
def parse_web_agent_response(response_text):
    """Parses the JSON response from the web agent and ensures it matches required format."""
    try:
        response_text = response_text.strip("```json").strip("```").strip()
        response_json = json.loads(response_text)
        if "link" in response_json and "summary" in response_json:
            return response_json
        else:
            raise ValueError("Response does not contain the required fields.")
    except json.JSONDecodeError:
        raise ValueError("Agent response is not valid JSON.")


# Web search tool using Tavily
web_search_tool = TavilySearchResults(
    max_results=3,
    include_answer=True,
    include_raw_content=True,
    search_depth="advanced",
    exclude_domains=["youtube.com"],
)

# Web agent
web_agent = create_react_agent(
    model=search_llm,
    tools=[web_search_tool],
    state_modifier=SystemMessage(content="""
    Your function is to perform web searches based on a query given to you. Given a 
    web query, you must perform searches with your tool and retrieve exactly **one** relevant, 
    **recent** article per query. 

    **Important:** Only return a **single, valid article per query** (not multiple articles or summary pages).

    **Rules to Follow Strictly:**
    - Only return **1 article per query**.
    - The article **must be from the last 7 days**.
    - **Do NOT return summary pages** like https://www.reuters.com/technology/ or https://www.nytimes.com/section/science/.
    - Ensure the article is **directly accessible** (not a paywalled summary).
    
    **Response Format (Must Be Valid JSON Only)**:
    ```json
    {
        "link": "link_for_topic",
        "summary": "summary_for_topic"
    }
    ```
    
    - **If a query does not return a valid article**, retry the search with adjusted keywords.
    - **Do NOT return empty lists.**
    - **Your response will be rejected if it does not have exactly 4 valid articles and summaries.**
    """)
)

# Web node with enforced 4 valid articles
def web_node(state: State):
    """
    Web search node responsible for gathering exactly 4 links and summaries based on search queries.
    """
    # Pulling state locally
    local_state = state.model_copy()

    # Looping through the topics and performing web search query 
    for i in range(len(local_state.web_queries)):
        valid_link: bool = False
        while not valid_link:
            local_state.messages = [HumanMessage(content=f"web query: {local_state.web_queries[i]}")]

            # Querying agent
            result = web_agent.invoke(local_state)
            new_message = result['messages'][-1].content

            # Parsing response
            parsed_output = parse_web_agent_response(new_message)

            # Validating link
            valid_link = 

        
        breakpoint()

    # TODO - continue with writing logic to replace general news links
    breakpoint()


# Defining the youtube agent


# Defining youtube node


# Defining the conditional edge router
def router(state: State):
    current_state = state.model_copy()
    if current_state.next == 'search':
        return 'web_node'   # ['web_node', 'youtube_node']
    elif current_state.next == '__end__':
        return '__end__'
    else:
        return '__end__'


builder = StateGraph(State)
builder.add_edge(START, "preparation_node")
builder.add_node("preparation_node", preparation_node)
builder.add_conditional_edges('preparation_node', router)
builder.add_node('web_node', web_node)
builder.add_edge('web_node', END)
graph = builder.compile()
    



async def invoke_agent():

    initial_state = State()

    result = graph.invoke(initial_state)

    breakpoint()

    return result
