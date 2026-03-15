from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from llms import writing_llm
from state import State

# title agent
title_agent = create_react_agent(
    model=writing_llm,
    tools=[],
    prompt="""
    Your function is to come up with creative article titles based on a summary.
    You will be given a summary of a web article or a youtube video.

    Ensure that your title is no more than 6 words. Make sure that the title is
    creative and properly summarizes the content.

    Return ONLY the title as a string - no explanations, no quotes, just the title text.
    """,
)


def title_node(state: State):
    """Title node responsible for taking in the 8 summaries and writing creative titles"""
    print("-----Running Title Node-----")
    # Pulling state locally
    local_state = state.model_copy()

    # Querying agent to write the titles for web summaries
    for i in range(len(local_state.web_final_summaries)):
        print("Writing title for: ", local_state.desired_subjects[i])
        local_state.messages = [
            (
                HumanMessage(
                    content=f"The following is the summary of a web article: {local_state.web_final_summaries[i]}"
                )
            )
        ]
        result = title_agent.invoke(local_state)
        new_message = result["messages"][-1].content
        local_state.web_final_titles.append(new_message)

    # Querying agent to write the titles for youtube summaries
    for i in range(len(local_state.youtube_final_summaries)):
        print("Writing title for: ", local_state.desired_subjects[i])
        local_state.messages = [
            (
                HumanMessage(
                    content=f"The following is the summary of a youtube video: {local_state.youtube_final_summaries[i]}"
                )
            )
        ]
        result = title_agent.invoke(local_state)
        new_message = result["messages"][-1].content
        local_state.youtube_final_titles.append(new_message)

    # Updating next in local state
    if (
        len(local_state.web_final_titles) > 0
        and len(local_state.youtube_final_titles) > 0
        and len(local_state.web_final_titles) == len(local_state.web_final_summaries)
        and len(local_state.youtube_final_titles) == len(local_state.youtube_final_summaries)
    ):
        local_state.next = "__end__"

    # Returning updated state and next node
    return local_state
