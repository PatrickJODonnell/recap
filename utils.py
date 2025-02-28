import asyncio
import httpx
import openai
import numpy as np
from langchain.document_loaders import WebBaseLoader

def get_openai_embedding(text: str):
    """Generates an embedding using OpenAI's `text-embedding-ada-002` model."""
    response = openai.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return np.array(response.data[0].embedding)

def cosine_similarity(vec1, vec2):
    """Computes cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def scrape_with_webbase(url: str) -> str:
    """
    Used to scrape content from a URL. Given a url it returns a string representation of the page content.
    """
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()
        return docs[0].page_content if docs else "No readable content found."
    except Exception:
        return None
    
def retry(max_retries=3, retry_delay=5):
    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    response = func(*args, **kwargs)
                    return response

                except httpx.WriteTimeout as e:
                    print(f"Retrying: {retries + 1} / {max_retries}")
                    print(f"https.WriteTimeout: {e}")

                asyncio.run(asyncio.sleep(retry_delay))
                retries += 1

            raise Exception(f"Failed to fetch data after {max_retries} retries")

        return wrapper

    return decorator