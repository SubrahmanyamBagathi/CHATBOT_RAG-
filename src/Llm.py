import os
from typing import Any, Dict, List
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()


def get_llm() -> ChatGroq:
    """Initialize and return the Groq LLM (reads GROQ_API_KEY from env)."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. Add it to your .env file or environment variables."
        )

    return ChatGroq(
        api_key=api_key,
        model_name="llama-3.1-8b-instant",
        temperature=0.1,
        max_tokens=1024,
    )


def rag_answer(query: str, retrieved_docs: List[Dict[str, Any]], llm: ChatGroq) -> str:
    """
    Generate an answer from retrieved context using the LLM.

    Args:
        query         : User question.
        retrieved_docs: List of dicts returned by RAGRetriever.retrieve().
        llm           : Initialized ChatGroq instance.

    Returns:
        Answer string.
    """
    context = "\n\n".join([doc["content"] for doc in retrieved_docs])

    if not context.strip():
        return "No relevant context found to answer this question."

    prompt = f"""Use the following context to answer the question concisely.

Context:
{context}

Question: {query}

Answer:"""

    response = llm.invoke(prompt)
    return response.content