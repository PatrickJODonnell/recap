import asyncio
import random
from typing import List
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
from prefect import flow, task
from utils import connectToFirestore
from langchain_core.messages import HumanMessage

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
@task
async def invoke_agent(user: dict):
    try: 
        topic_length = len(user['interests'])
        desired_subjects: List[str] = []
        if topic_length >= 3:
            desired_subjects = random.sample(user['interests'], 3)
        else:
            desired_subjects = user['interests']
        initial_state = State()
        populated_state = initial_state.model_copy(update={"desired_subjects": desired_subjects, "user": user['id']})
        populated_state.messages.append(HumanMessage(content=f"desired_subjects: {desired_subjects}"))
        result = await graph.ainvoke(populated_state)
        return result
    except Exception as e:
        print('ERROR', e)
        return None


# Function for pulling users and kicking off runs
@flow(log_prints=True)
async def process_users():
    db = await connectToFirestore()
    users = db.collection('Users')
    last_doc = None
    while True:
        query = users.order_by("signUpDate").limit(100)

        if last_doc:
            query = query.start_after(last_doc)  # Using start_after for pagination

        docs = query.stream()
        docs_list = list(docs)  # Converting to list to check length

        if not docs_list:  # If no more documents, stop
            break

        last_doc = docs_list[-1]  # Set last document for next batch

        tasks = []
        for doc in docs_list:
            user_obj = doc.to_dict()
            user_obj["id"] = doc.id
            tasks.append(invoke_agent(user_obj))

        results = await asyncio.gather(*tasks)
        breakpoint()
        # TODO - PARSE RESULTS AND ADD THEM TO DB
    breakpoint()

