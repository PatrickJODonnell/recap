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
        web_query = f'Find a recent article about {topic} related news.'
        youtube_query = f'Find a recent video about {topic} related news.'
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


# Defining the web agent
web_search_tool = TavilySearchResults(
            max_results=5,
            include_answer=True,
            include_raw_content=True,
            search_depth="advanced",
            # include_domains = []
            exclude_domains = ['youtube.com']
        )

web_agent = create_react_agent(
    model=search_llm, tools=[web_search_tool], state_modifier=SystemMessage(content="""
    Your function is to perform web searches based on queries given to you. Given a list of 
    web queries you should use these to perfom searches with your tool. You should gather both a list of                                                                                                     
    links and summaries of each web article that you find. Remember, only find 1 article per query. 
                                                      
    For example, if 'Find a recent article about Philadelphia 76ers related news' is the web query you are
    given, you can find a article such as this https://www.si.com/nba/76ers/news/reggie-jackson-expected-to-find-new-team-after-76ers-wizards-trade-01jke4qnbz5a
    and provide a summary on the article that gives the main point. Make sure to look into an actual article,
    not just a summary page of articles. A summary page of articles looks like this: https://www.reuters.com/technology/
    where the page has links to many other articles.                                                                          
    Finally, ensure that the news articles that you return have been recently posted in the last 7 days. If the article 
    you have chosen is older than 7 days old find a different article.                                                                                                                                                                                                                                                                                                                                                                                       

    Your response should follow this strict format: LINKS: [link_for_topic_1, link_for_topic_2, link_for_topic_3, link_for_topic_4] AND
    SUMMARIES: [summary_for_topic_1, summary_for_topic_1, summary_for_topic_1, summary_for_topic_1]. Each element in the provided 
    arrays should be replaced with the actual links and summaries that you generate.                                                                        
    """)
)

# Defining web node
def web_node(state: State):
    """
    Web search node responsible for gathering links and generating summaries based on search queries. 
    """
    # Pulling state down locally and altering its messages for the current agent
    local_state = state.model_copy()
    local_state.messages = [HumanMessage(content=f"web queries: {local_state.web_queries}")]

    # Generating links and summaries
    result = web_agent.invoke(local_state)
    new_message = result['messages'][-1].content
    print(new_message)
    print(local_state.desired_subjects)
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
