from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from llms import writing_llm
from state import State

# proofing agent
proofing_agent = create_react_agent(
    model=writing_llm,
    tools=[],
    state_modifier=SystemMessage(
        content="""
    Your function is to proofread an excerpt. You should correct any grammatical
    mistakes that you find and also make the writing sound more human like. 
                                 
    You will be given a paragraph written by a different agent. Take this excerpt and
    make the writing sound more casual and natural (for example, replace any words that 
    aren't typically used in everyday conversation).
                                 
    Ensure that you return a result that is still around 5 sentences long.

    Return your summary as a string.                                                          
    """
    ),
)


def proofing_node(state: State):
    """Proofing node responsible for revising the 8 excerpts to make them sound more natural."""
    print("-----Running Proofing Node-----")
    # Pulling state locally
    local_state = state.model_copy()

    # Querying agent to re-write newsletter like excerpts for articles
    for i in range(len(local_state.web_final_summaries)):
        print("Proofing article newsletter for: ", local_state.desired_subjects[i])
        local_state.messages = [
            (HumanMessage(content=f"The following is the excerpt to edit: {local_state.web_final_summaries[i]}"))
        ]
        result = proofing_agent.invoke(local_state)
        new_message = result["messages"][-1].content
        local_state.web_final_summaries[i] = new_message

    # Querying agent to re-write newsletter like excerpts for youtube videos
    for i in range(len(local_state.youtube_final_summaries)):
        print("Proofing video newsletter for: ", local_state.desired_subjects[i])
        local_state.messages = [
            (HumanMessage(content=f"The following is the excerpt to edit: {local_state.youtube_final_summaries[i]}"))
        ]
        result = proofing_agent.invoke(local_state)
        new_message = result["messages"][-1].content
        local_state.youtube_final_summaries[i] = new_message

    # Updating next in local state
    if (
        len(local_state.web_final_summaries) > 0
        and len(local_state.youtube_final_summaries) > 0
        and len(local_state.web_final_summaries) == len(local_state.web_links)
        and len(local_state.youtube_final_summaries) == len(local_state.youtube_links)
    ):
        local_state.next = "title"
    else:
        local_state.next = "__end__"

    # Returning updated state and next node
    return local_state
