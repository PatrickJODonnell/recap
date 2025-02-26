from pydantic import BaseModel
from langchain_core.messages import BaseMessage
from typing import List

# Defining the state dict
class State(BaseModel):
    # For Preparation
    messages: List[BaseMessage] = []
    desired_subjects: List[str] = []
    web_queries: List[str] = []
    youtube_queries: List[str] = []
    # For Search
    web_links: List[str] = []
    youtube_links: List[str] = []
    web_summaries: List[str] = []
    youtube_summaries: List[str] = []
    # For Write / Proof
    web_final_summaries: List[str] = []
    youtube_final_summaries: List[str] = []
    # For All
    next: str = None