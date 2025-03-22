from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# Loading env variables
load_dotenv(override=True)

# All llms for agents
llm = ChatOpenAI(model = "gpt-4o", temperature = 1.0, api_key=os.getenv('OPENAI_API_KEY'))
search_llm = ChatOpenAI(model = "gpt-4o", temperature = 0.0, api_key=os.getenv('OPENAI_API_KEY'))
writing_llm = ChatOpenAI(model = "gpt-4o", temperature = 0.9, api_key=os.getenv('OPENAI_API_KEY'))