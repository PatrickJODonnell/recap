from langgraph.graph import StateGraph, START, END
import os
from dotenv import load_dotenv
import openai
from preparation_agent import preparation_node
from proofing_agent import proofing_node
from web_agent import web_node
from youtube_agent import youtube_node
from writer_agent import writer_node
from state import State

# Loading env variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


# Defining the conditional edge router
def router(state: State):
    current_state = state.model_copy()
    if current_state.next == 'web':
        return 'web_node'
    elif current_state.next == 'video':
        return 'youtube_node'
    elif current_state.next == 'write':
        return 'writer_node'
    elif current_state.next == 'proof':
        return 'proofing_node'
    elif current_state.next == '__end__':
        return '__end__'
    else:
        return '__end__'


# Defining graph
builder = StateGraph(State)
builder.add_edge(START, "preparation_node")
builder.add_node("preparation_node", preparation_node)
builder.add_conditional_edges('preparation_node', router)
builder.add_node('web_node', web_node)
builder.add_conditional_edges('web_node', router)
builder.add_node('youtube_node', youtube_node)
builder.add_conditional_edges('youtube_node', router)
builder.add_node('writer_node', writer_node)
builder.add_conditional_edges('writer_node', router)
builder.add_node('proofing_node', proofing_node)
builder.add_edge('proofing_node', END)
graph = builder.compile()

# Function call for running agent
async def invoke_agent():
    initial_state = State()
    result = graph.invoke(initial_state)
    # TODO - Write this stuff off to a database
    breakpoint()
    return result
