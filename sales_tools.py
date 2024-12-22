# Imports
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools import tool


search_tool = TavilySearchResults(
            max_results=10,
            include_answer=True,
            include_raw_content=True,
            include_images=True,
            search_depth="advanced",
            # include_domains = []
            # exclude_domains = []
        )

