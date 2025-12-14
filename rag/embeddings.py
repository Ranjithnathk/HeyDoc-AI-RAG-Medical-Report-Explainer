import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Returns embeddings for a list of texts.
    """
    if not texts:
        return []
    resp = _client.embeddings.create(model=_EMBED_MODEL, input=texts)
    return [item.embedding for item in resp.data]
