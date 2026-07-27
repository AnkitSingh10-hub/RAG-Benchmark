import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

AZURE_ENDPOINT = (
    "https://ankitsinghtheweeknd691-6608-reso.services.ai.azure.com/openai/v1"
)

EMBEDDING_MODEL = "text-embedding-3-large"

# Azure AI Foundry client - embeddings
embedding_client = OpenAI(
    base_url=AZURE_ENDPOINT,
    api_key=os.getenv("AZURE_FOUNDRY_API_KEY"),
)


def embed_texts(
    texts: list[str],
    batch_size: int = 32,
) -> list[list[float]]:

    vectors = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        response = embedding_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
        )

        vectors.extend(item.embedding for item in response.data)

    return vectors


def embed_query(text: str) -> list[float]:

    response = embedding_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )

    return response.data[0].embedding
