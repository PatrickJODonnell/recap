from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage
from prefect import task
from llms import writing_llm
from state import State

# writer agent
writer_agent = create_react_agent(
    model=writing_llm,
    tools=[],
    state_modifier=SystemMessage(content="""
    Your function is to write descriptive and creative one-paragraph                              
    excepts. You will be given either the summary of a web article or the 
    summary of a youtube video.

    Ensure that your excerpt is no more than 5 sentences. Make sure to highlight key points
    and make your writing as interesting as possible.
                                 
    Try your best to make the excerpt interesting and clever. You may include some humor too,
    but don't overdo it. Try to sound natural and human-like.
                                 
    Avoid using pointless similies and metaphors.

    Return your summary as a string.                                                          
    """)
)

def writer_node(state: State):
    """Writer node responsible for taking in the 8 summaries and making them into newletter like paragraphs"""
    print('-----Running Writer Node-----')
    # Pulling state locally
    local_state = state.model_copy()

    # Querying agent to re-write newsletter like excerpts for articles
    for i in range(len(local_state.web_summaries)):
        print('Writing article newsletter for: ', local_state.desired_subjects[i])
        local_state.messages = [(HumanMessage(content=f'The following is the summary of a web article: {local_state.web_summaries[i]}'))]
        result = writer_agent.invoke(local_state)
        new_message = result['messages'][-1].content
        local_state.web_final_summaries.append(new_message)
    
    # Querying agent to re-write newsletter like excerpts for youtube videos
    for i in range(len(local_state.youtube_summaries)):
        print('Writing video newsletter for: ', local_state.desired_subjects[i])
        local_state.messages = [(HumanMessage(content=f'The following is the summary of a youtube video: {local_state.youtube_summaries[i]}'))]
        result = writer_agent.invoke(local_state)
        new_message = result['messages'][-1].content
        local_state.youtube_final_summaries.append(new_message)

    # Updating next in local state
    if (len(local_state.web_final_summaries) > 0 and len(local_state.youtube_final_summaries) > 0 and len(local_state.web_final_summaries) == len(local_state.web_links) and len(local_state.youtube_final_summaries) == len(local_state.youtube_links)):
        local_state.next = "proof"
    else:
        local_state.next = "__end__"

    # Returning updated state and next node
    return local_state
