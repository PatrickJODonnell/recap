from typing import Literal, List
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START

llm = ChatOpenAI(model = "gpt-4o", temperature = 0.0)

# Defining the state dict and the router
class State(TypedDict):
    desired_subjects = List[str]
    web_query: str
    youtube_query: str
    next: str

members = ["preparation_agent", "web_agent", "youtube_agent", "writing_agent", "proof_reading_agent"]
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
)





async def invoke_agent(industry: str, conditions: str):
    result = graph.invoke({
        "input_industry": industry,
        "input_condition": conditions
    })

    return result
