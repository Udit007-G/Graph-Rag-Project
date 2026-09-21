import os
import time
from dotenv import load_dotenv

load_dotenv()

TG_HOST = os.getenv("TG_HOST", "")
TG_TOKEN = os.getenv("TG_TOKEN", "")
TG_GRAPHNAME = os.getenv("TG_GRAPHNAME", "olympics")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

CORPUS_PATH = os.path.join(os.path.dirname(__file__), "..", "corpus", "corpus.jsonl")
EVAL_PUBLIC_PATH = os.path.join(os.path.dirname(__file__), "..", "questions", "eval_public.jsonl")
EVAL_HIDDEN_PATH = os.path.join(os.path.dirname(__file__), "..", "questions", "eval_hidden.jsonl")

MAX_STEPS = 6
MAX_TOKENS_PER_QUERY = 4000
TEMPERATURE = 0.0
MAX_RETRIES = 3
RETRY_DELAY = 10


def get_openai_client():
    """Lazy-init OpenAI-compatible client (Groq, Gemini, or OpenAI)."""
    from openai import OpenAI

    if LLM_PROVIDER == "groq" and GROQ_API_KEY:
        return OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)
    elif LLM_PROVIDER == "gemini" and GEMINI_API_KEY:
        return OpenAI(api_key=GEMINI_API_KEY, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
    elif OPENAI_API_KEY:
        return OpenAI(api_key=OPENAI_API_KEY)
    else:
        raise ValueError("No LLM API key configured. Set GROQ_API_KEY, GEMINI_API_KEY, or OPENAI_API_KEY in .env")


def get_model_name():
    """Get the model name for the configured provider."""
    if LLM_PROVIDER == "groq" and GROQ_API_KEY:
        return GROQ_MODEL
    elif LLM_PROVIDER == "gemini" and GEMINI_API_KEY:
        return GEMINI_MODEL
    return OPENAI_MODEL


def llm_completion(messages, temperature=0.0, max_tokens=500, response_format=None):
    """LLM completion with retry logic for rate limiting."""
    client = get_openai_client()
    model = get_model_name()

    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format:
        kwargs["response_format"] = response_format

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(**kwargs)
            return response
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "rate_limit" in error_msg.lower():
                wait_time = RETRY_DELAY * (attempt + 1)
                print(f"  [Rate limit] Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                raise

    raise Exception("Max retries exceeded due to rate limiting")
