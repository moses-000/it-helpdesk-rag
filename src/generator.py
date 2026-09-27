"""Step 3 of RAG: ask an LLM to answer using ONLY the retrieved chunks.

Grounding the model in retrieved text is what makes RAG useful: answers follow
company policy instead of generic internet advice, and each claim can be traced
back to a cited source.
"""
from openai import OpenAI

from . import config

SYSTEM_PROMPT = """You are the IT Service Desk assistant for Harrowgate Logistics.
Answer the employee's question using ONLY the numbered knowledge base extracts provided.

Rules:
- Give clear, practical steps. Keep answers short (under 150 words) unless steps require more.
- Cite the extracts you used in square brackets, e.g. [1] or [2][3].
- If the extracts do not contain the answer, say you don't have that information and
  suggest raising a ticket at the IT portal or calling the Service Desk on extension 4400.
  Never invent policies, phone numbers, links or settings.
- Never ask the user for their password."""


def format_context(chunks: list[dict]) -> str:
    return "\n\n".join(f"[{i}] {c['text']}" for i, c in enumerate(chunks, start=1))


def build_messages(question: str, chunks: list[dict], history: list[dict] | None = None) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += (history or [])[-4:]  # short memory for follow-up questions
    messages.append({"role": "user",
                     "content": f"Knowledge base extracts:\n{format_context(chunks)}\n\nQuestion: {question}"})
    return messages


def llm_available() -> bool:
    return bool(config.LLM_API_KEY) or "localhost" in config.LLM_BASE_URL


def generate_answer(question: str, chunks: list[dict], history: list[dict] | None = None) -> str:
    client = OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY or "ollama")
    response = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=build_messages(question, chunks, history),
        temperature=0.1,  # low temperature = consistent, factual answers
    )
    return response.choices[0].message.content.strip()
