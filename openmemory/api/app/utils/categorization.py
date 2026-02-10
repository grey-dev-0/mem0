import json
import logging
import os
from typing import List

from app.utils.prompts import MEMORY_CATEGORIZATION_PROMPT
from dotenv import load_dotenv
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

# Optional OpenAI client for fallback
try:
    from openai import OpenAI
    openai_client = OpenAI()
except Exception:
    openai_client = None

load_dotenv()


class MemoryCategories(BaseModel):
    categories: List[str]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15))
def get_categories_for_memory(memory: str) -> List[str]:
    """Get categories for a memory using Ollama LLM with OpenAI fallback."""
    messages = [
        {"role": "system", "content": MEMORY_CATEGORIZATION_PROMPT + "\nReturn only JSON: {\"categories\": [\"...\"]}"},
        {"role": "user", "content": memory}
    ]

    # 1) Try mem0/ollama configured LLM first
    try:
        from app.utils.memory import get_memory_client
        mc = get_memory_client()
        if mc and getattr(mc, 'llm', None):
            resp = mc.llm.generate_response(messages, response_format="json")
            content = resp.get("content") if isinstance(resp, dict) else resp
            data = json.loads(content) if isinstance(content, str) else content
            cats = data.get("categories", []) if isinstance(data, dict) else []
            return [str(c).strip().lower() for c in cats if str(c).strip()]
    except Exception as e:
        logging.error(f"[ERROR] Ollama categorization failed: {e}")

    # 2) Fallback to OpenAI if key is present
    try:
        if os.getenv("OPENAI_API_KEY") and openai_client is not None:
            completion = openai_client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=messages,
                response_format=MemoryCategories,
                temperature=0
            )
            parsed: MemoryCategories = completion.choices[0].message.parsed
            return [cat.strip().lower() for cat in parsed.categories]
    except Exception as e:
        logging.error(f"[ERROR] OpenAI fallback categorization failed: {e}")

    # 3) Last resort: no categories
    return []
