from langchain_openai import ChatOpenAI

# All llms for agents
llm = ChatOpenAI(model = "gpt-4o", temperature = 1.0)
search_llm = ChatOpenAI(model = "gpt-4o", temperature = 0.0)
writing_llm = ChatOpenAI(model = "gpt-4o", temperature = 1.0)