from typing import Literal, List
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START
from agent_types import CompanyResult, EmployeeResult

llm = ChatOpenAI(model = "gpt-4o", temperature = 0.0)

class State(TypedDict):
    input_industry: str
    input_condition: str
    query: str
    company_links = List[str]
    companies = List[CompanyResult]
    is_companies_done = bool
    employee_links = List[str]
    employees = List[EmployeeResult]
    is_employees_done = bool
    next_node: str

def router(state) -> Literal["generate_search_query", "company_search", "company_scrape", "company_verification", "employee_scrape", "employee_verification", "__end__"]:
    """
    Node router used to send the graph to its next state.
    """
    return state['next_node'] 

def generate_search_query(state) -> str:
    """
    Taking in user inputs and queryling the llm to create a search friendly query string.
    Returns the next node for the conditional edges.
    """

def company_search(state) -> str:
    """
    Taking in the search query and performing a web search. Here we will begin to populate the company_links list
    Returns the next node for the conditional edges.
    """

def company_scrape(state) -> str:
    """
    Taking in the list of company links and will perform web scrapes to fill in the companies list.
    Returns the next node for the conditional edges. 
    """

def company_verification(state) -> str:
    """
    Taking in the list of company objects and performing verification tests prior to searching employees.
    Returns the next node for conditional edges.
    """

def employee_scrape(state) -> str:
    """
    Taking in the companies list and pulling information for all active employees found. We will populate the 
    employees list here.
    Returns the next node for conditional edges.
    """

def employee_verification(state) -> str:
    """
    Taking in the list of employee objects and performs verification tests prior to returning to the user.
    Returns the next node for conditional edges.
    """

# Build graph
builder = StateGraph(State)
builder.add_node("generate_search_query", generate_search_query)
builder.add_node("company_search", company_search)
builder.add_node("company_scrape", company_scrape)
builder.add_node("company_verification", company_verification)
builder.add_node("employee_scrape", employee_scrape)
builder.add_node("employee_verification", employee_verification)

# Logic
builder.add_edge(START, "generate_search_query")
builder.add_conditional_edges("generate_search_query", router)
builder.add_conditional_edges("company_search", router)
builder.add_conditional_edges("company_scrape", router)
builder.add_conditional_edges("company_verification", router)
builder.add_conditional_edges("employee_scrape", router)
builder.add_conditional_edges("employee_verification", router)

# Add
graph = builder.compile()

def invoke_graph(industry: str, conditions: str):
    result = graph.invoke({
        "input_industry": industry,
        "input_condition": conditions
    })

    return result
