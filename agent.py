from typing import Literal, List
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from temp_obj import topics
import random
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage

llm = ChatOpenAI(model = "gpt-4o", temperature = 1.0)


# Defining the state dict and the router
class State(TypedDict):
    desired_subjects = List[str]
    web_queries: List[str]
    youtube_queries: List[str]
    next: str

members = ["preparation_agent"] # , "web_agent", "youtube_agent", "writing_agent", "proof_reading_agent"
options = members + ["FINISH"]
class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH."""
    next: Literal[*options] # type: ignore


# Defining system prompt
system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    f" following workers: {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. The workers must"
    " act in the order preparation_agent, web_agent, youtube_agent,"
    " writing_agent, proof_reading_agent. Each agent must finish their task"
    " before the next agent begins working. When finished, respond with FINISH."
    " For now, we only have the preparation_agent so just perform that step!"
)


# Defining tools for the agents



# Defining supervisor node
def supervisor_node(state: State) -> Command[Literal[*members, "__end__"]]: # type: ignore
    """
    Supervisor node responsible for calling upon other nodes.
    """
    messages = [{"role": "system", "content": system_prompt},] + [
        {"role": "user", "content": "Please find and summarize an internet article and a Youtube video for each one of the following topics and return it in a newletter format."}
        ]
    response = llm.with_structured_output(Router).invoke(messages)
    goto = response["next"]
    if goto == "FINISH":
        goto = END
    return Command(goto=goto, update={"next": goto})


# Defining the preparation agent
preparation_agent = create_react_agent(
    model=llm, tools=[], state_modifier=SystemMessage(content="""
    Your function is to suggest new topics of interest. To do this, take in a list of topics and determine if they are appropriate.
    If any are not appropriate, replace that topic. Your final list should consist of 4 topics. If the length of your topic list is 
    less than 4, you should add more topics that the user may be interested in until the length of the topic list is 4.

    Your response should start with: TOPICS: [topic1, topic2, topic3, topic4].
    """)
)


# Defining preparation node
def preparation_node(state: State) -> Command[Literal["supervisor"]]:
    """
    Preparation node responsible for gathering topics, suggesting a topic, determining if topics are appropriate, and return search queries
    """
    # Pulling topics from database (for now just pulling from a dict)
    topic_length = len(topics)
    desired_subjects = []
    if topic_length >= 3:
        desired_subjects = random.sample(topics, 3)
    else:
        state["desired_subjects"] = topics

    # Updating state

# TODO - pick up here
    # Generating suggested topics and replacing innapropriate topics
    result = preparation_agent.invoke(state)
    print(result)



async def invoke_agent():
    result = preparation_node(state=State)

    breakpoint()

    return result
