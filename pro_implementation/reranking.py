"""Reranking strategies for the RAG pipeline.

Two interchangeable rerankers are provided here:

- rerank_llm     — the original approach: a general chat LLM (gpt-5.6-luna
                    via Azure) is shown all candidate chunks at once and
                    asked to return a strict JSON ordering of them.
- rerank_cohere  — Cohere Rerank v4 (a purpose-built cross-encoder),
                    deployed as a model inside the same Azure AI Foundry
                    resource, called directly via requests.

Both take (question, chunks) and return chunks reordered most-to-least
relevant. answer.py imports whichever one it wants to use - see the
RERANKER switch at the top of that file.
"""

import os

import requests
from openai import OpenAI
from pydantic import BaseModel, Field

from .models import Result

AZURE_ENDPOINT = (
    "https://ankitsinghtheweeknd691-6608-reso.services.ai.azure.com/openai/v1"
)

# --- rerank_llm config --------------------------------------------------

DEFAULT_MODEL = "gpt-5.6-luna"

llm_client = OpenAI(
    base_url=AZURE_ENDPOINT,
    api_key=os.getenv("AZURE_FOUNDRY_API_KEY"),
)


class RankOrder(BaseModel):
    order: list[int] = Field(
        description="The order of relevance of chunks, from most relevant to least relevant, by chunk id number"
    )


# --- rerank_cohere config -------------------------------------------------

# Cohere Rerank v4, deployed as a model inside this same Azure AI Foundry
# resource (not a separate Cohere account) - reuses AZURE_FOUNDRY_API_KEY.
COHERE_RERANK_URL = "https://ankitsinghtheweeknd691-6608-reso.services.ai.azure.com/providers/cohere/v2/rerank"
# IMPORTANT: must exactly match your deployment name in Azure AI Foundry -
# check the "Get endpoint" page for that deployment if this 404s/500s.
COHERE_RERANK_MODEL = "Cohere-rerank-v4.0-pro"


# --- rerank_llm -----------------------------------------------------------


def rerank_llm(question: str, chunks: list[Result]) -> list[Result]:
    """LLM-based rerank (gpt-5.6-luna). Returns ALL chunks reordered
    (not truncated) - caller is expected to slice to top_n if needed.
    """
    system_prompt = """
You are a document re-ranker.
You are provided with a question and a list of relevant chunks of text from a query of a knowledge base.
The chunks are provided in the order they were retrieved; this should be approximately ordered by relevance, but you may be able to improve on that.
You must rank order the provided chunks by relevance to the question, with the most relevant chunk first.
Reply only with the list of ranked chunk ids, nothing else. Include all the chunk ids you are provided with, reranked.
"""
    user_prompt = f"The user has asked the following question:\n\n{question}\n\nOrder all the chunks of text by relevance to the question, from most relevant to least relevant. Include all the chunk ids you are provided with, reranked.\n\n"
    user_prompt += "Here are the chunks:\n\n"
    for index, chunk in enumerate(chunks):
        user_prompt += f"# CHUNK ID: {index + 1}:\n\n{chunk.page_content}\n\n"
    user_prompt += "Reply only with the list of ranked chunk ids, nothing else."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    schema = RankOrder.model_json_schema()
    schema["additionalProperties"] = False

    response = llm_client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=messages,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "RankOrder",
                "schema": schema,
                "strict": True,
            },
        },
    )
    reply = response.choices[0].message.content
    order = RankOrder.model_validate_json(reply).order
    print("=" * 80)
    print(question)
    print("Expected:", len(chunks))
    print("Returned:", order)
    print("=" * 80)
    return [chunks[i - 1] for i in order]


# --- rerank_cohere ----------------------------------------------------


def rerank_cohere(
    question: str, chunks: list[Result], top_n: int | None = None
) -> list[Result]:
    """Cohere Rerank v4 (cross-encoder), deployed on Azure AI Foundry.
    Called directly via requests (not the cohere SDK) since the SDK is
    known to double-append /v2/rerank onto Azure Foundry base URLs.

    top_n: how many of the reranked chunks to ask Cohere to return.
    Defaults to all chunks (matching rerank_llm's behavior) if not given.
    """
    if top_n is None:
        top_n = len(chunks)

    documents = [chunk.page_content for chunk in chunks]

    response = requests.post(
        COHERE_RERANK_URL,
        headers={
            "api-key": os.getenv("AZURE_FOUNDRY_API_KEY"),
            "Content-Type": "application/json",
        },
        json={
            "model": COHERE_RERANK_MODEL,
            "query": question,
            "documents": documents,
            "top_n": min(top_n, len(chunks)),
        },
    )
    response.raise_for_status()
    data = response.json()

    reranked = [chunks[result["index"]] for result in data["results"]]
    print("=" * 80)
    print(question)
    print("Expected:", len(chunks))
    print("Returned (Cohere indices):", [r["index"] for r in data["results"]])
    print("=" * 80)
    return reranked
