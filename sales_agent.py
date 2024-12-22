# Imports
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from sales_tools import search_tool
from langchain_core.prompts import ChatPromptTemplate
import uuid


# Instantiating memory persistence
memory = MemorySaver()

# Instantiating the LLM
model = ChatOpenAI(
    model="gpt-4o",
    temperature=0.2,
    max_tokens=5000,
    timeout=300,
    max_retries=2)

# Creating tools list 
tools = [search_tool]

# Setting up a chat prompt template for the agent
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a sales expert who has over 20 years of experience in selling products and services for businesses."),
    ("placeholder", "{messages}"),
    ("user", "Remember, be concise with your answers and abide by the given format!"),
])

# Creating agent
agent = create_react_agent(model=model, tools=tools, checkpointer=memory, state_modifier=prompt)

# Testing function to call 
def call_agent(input_obj: dict):
    """
    Function used to invoke the agent for testing.
    Will need to work out how to make this a callable later.
    """
    # Setting up thread id for the checkpointer
    thread_id = uuid.uuid4()
    config = {"configurable": {"thread_id": f"{thread_id}"}}

    # Setting up messages based on gathered user input
    inputs = {"messages": [(
        "user", 
        f"""I am a salesman and I sell {input_obj.get('product')}.

        My goal is to find companies in the {input_obj.get('target_industry')} industry that I can sell to.

        The companies must fit this condition: {input_obj.get('condition')}

        Search for 10 companies that you believe would be good clients to sell to.

        Please make sure to respond with a list of objects that follows the following format as a json object.

        {{
            'company_name': (name of the company),
            'company_logo': (the logo of the company),
            'company_location': (geographical location of the company),
            'company_summary': (five words or less summary of what the company does)
            'company_url': (url of the company website),
            'company_linkedin_url' : (url of the company's linkedin page)
        }}

        If the LinkedIn page contains "This LinkedIn Page isn\'t available", then please return None instead.
        """
    )]}

    # Invoking the agent
    response = agent.invoke(input=inputs, config=config)

    # Returning the response
    return response, thread_id
