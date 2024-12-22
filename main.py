# Imports
import os
from dotenv import load_dotenv
from sales_agent import call_agent

# Loading env variables
load_dotenv()

# Setting up LLM model
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Gathering user inputs for the company search
product: str = input('What do you sell: ')
target_industry: str =input('What Industry are you looking to sell to: ')
conditions: str = input('Are there any conditions surrounding your sales search: ')

# Starting flow
response, thread_id = call_agent({
    'product': product,
    'target_industry': target_industry,
    'conditions': conditions
})

print(response)

breakpoint()
