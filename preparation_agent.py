from llms import llm
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage
from state import State
import re
import ast
from temp_obj import topics
from typing import List
import random
from utils import connectToFirestore

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
async def preparation_node(state: State):
    """
    Preparation node responsible for gathering topics, suggesting a topic, determining if topics are appropriate, and return search queries.
    """
    # Pulling topics from database (for now just pulling from a dict)
    print('-----Running Preparation Node-----')
    db = await connectToFirestore()
    stuff = db.collection('Users')
    breakpoint()
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
        youtube_query = f'site:youtube.com {topic}'
        updated_state.web_queries.append(web_query)
        updated_state.youtube_queries.append(youtube_query)

    # Updating message and next in local state
    updated_state.messages.append(SystemMessage(content=f"Suggested topics: {result}"))
    if (len(updated_state.desired_subjects) == 4 and len(updated_state.web_queries) == 4 and len(updated_state.youtube_queries) == 4):
        updated_state.next = "web"
    else:
        updated_state.next = "__end__"

    # Returning updated state and next node
    print('Chosen Topics: ', updated_state.desired_subjects)
    return updated_state