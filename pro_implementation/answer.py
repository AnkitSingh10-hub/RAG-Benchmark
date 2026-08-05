import os
from pathlib import Path

from chromadb import PersistentClient
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

from .embeddings import embed_query
from .models import Result
from .reranking import rerank_llm, rerank_cohere

load_dotenv(override=True)

DB_NAME = str(Path(__file__).parent / "preprocessed_db")
COLLECTION_NAME = "docs"  # must match ingest.py exactly

AZURE_ENDPOINT = (
    "https://ankitsinghtheweeknd691-6608-reso.services.ai.azure.com/openai/v1"
)

# Generation/rerank/query-rewrite model — Azure AI Foundry (gpt-5.6-luna).
DEFAULT_MODEL = "gpt-5.6-luna"

RETRIEVAL_K = 25  # wider net for the embedding search
RERANK_TOP_N = 5  # what actually reaches the RAG prompt

# Which reranker to use - just point this at rerank_llm or rerank_cohere.
# Both live in reranking.py and share the same (question, chunks) -> chunks signature.
RERANKER = rerank_cohere


# Azure AI Foundry client - reranking, query rewriting, RAG answers
client = OpenAI(
    base_url=AZURE_ENDPOINT,
    api_key=os.getenv("AZURE_FOUNDRY_API_KEY"),
)

chroma = PersistentClient(path=DB_NAME)

collection = chroma.get_collection(COLLECTION_NAME)


SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
Your answer will be evaluated for accuracy, relevance and completeness, so make sure it only answers the question and fully answers it.
If you don't know the answer, say so.
For context, here are specific extracts from the Knowledge Base that might be directly relevant to the user's question:
{context}

With this context, please answer the user's question. Be accurate, relevant and complete.
"""


def fetch_context_unranked(question: str) -> list[Result]:
    query_embedding = embed_query(question)
    results = collection.query(
        query_embeddings=[query_embedding], n_results=RETRIEVAL_K
    )
    chunks = []
    for document, metadata in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append(Result(page_content=document, metadata=metadata))
    return chunks


def fetch_context(question: str) -> list[Result]:
    chunks = fetch_context_unranked(question)
    reranked = RERANKER(question, chunks)
    return reranked[:RERANK_TOP_N]


def make_rag_message(
    question: str, history: list[dict], chunks: list[Result]
) -> list[dict]:
    """Builds the full messages list (system + history + new user question)
    given pre-retrieved chunks for a RAG-augmented chat completion call.
    """
    context = "\n\n".join(
        f"# Source: {chunk.metadata.get('source', 'unknown')}\n{chunk.page_content}"
        for chunk in chunks
    )

    system_message = {
        "role": "system",
        "content": SYSTEM_PROMPT.format(context=context),
    }

    messages = [system_message] + history + [{"role": "user", "content": question}]
    return messages


def rewrite_query(question: str, history: list[dict] = []) -> str:
    """Rewrite the user's question to be a more specific question that is
    more likely to surface relevant content in the Knowledge Base.
    """
    message = f"""
You are in a conversation with a user, answering questions about the company Insurellm.
You are about to look up information in a Knowledge Base to answer the user's question.

This is the history of your conversation so far with the user:
{history}

And this is the user's current question:
{question}

Respond only with a single, refined question that you will use to search the Knowledge Base.
It should be a VERY short specific question most likely to surface content. Focus on the question details.
Don't mention the company name unless it's a general question about the company.
IMPORTANT: Respond ONLY with the knowledgebase query, nothing else.
"""
    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[{"role": "system", "content": message}],
    )
    return response.choices[0].message.content


def answer_question(
    question: str,
    history: list[dict] = [],
) -> tuple[str, list[Result]]:
    """Answer a question using RAG and return the answer and the retrieved context."""
    query = rewrite_query(question, history)
    chunks = fetch_context(query)
    messages = make_rag_message(question, history, chunks)
    response = client.chat.completions.create(model=DEFAULT_MODEL, messages=messages)
    return response.choices[0].message.content, chunks
