import os
import time
from dotenv import load_dotenv

load_dotenv()

TG_HOST = os.getenv("TG_HOST", "")
TG_TOKEN = os.getenv("TG_TOKEN", "")
TG_GRAPHNAME = os.getenv("TG_GRAPHNAME", "olympics")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")

# Support multiple keys (comma-separated) for fallback
GROQ_API_KEYS = [k.strip() for k in os.getenv("GROQ_API_KEY", "").split(",") if k.strip()]
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

GEMINI_API_KEYS = [k.strip() for k in os.getenv("GEMINI_API_KEY", "").split(",") if k.strip()]
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

OPENAI_API_KEYS = [k.strip() for k in os.getenv("OPENAI_API_KEY", "").split(",") if k.strip()]
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

CORPUS_PATH = os.path.join(os.path.dirname(__file__), "..", "corpus", "corpus.jsonl")
EVAL_PUBLIC_PATH = os.path.join(os.path.dirname(__file__), "..", "questions", "eval_public.jsonl")
EVAL_HIDDEN_PATH = os.path.join(os.path.dirname(__file__), "..", "questions", "eval_hidden.jsonl")

MAX_STEPS = 6
MAX_TOKENS_PER_QUERY = 4000
TEMPERATURE = 0.0
MAX_RETRIES = 3
RETRY_DELAY = 15  # Increased for rate limit handling

_key_indices = {"groq": 0, "gemini": 0, "openai": 0}


def _get_keys(provider: str):
    if provider == "groq":
        return GROQ_API_KEYS
    elif provider == "gemini":
        return GEMINI_API_KEYS
    elif provider == "openai":
        return OPENAI_API_KEYS
    return []


def _get_base_url(provider: str):
    if provider == "groq":
        return GROQ_BASE_URL
    elif provider == "gemini":
        return "https://generativelanguage.googleapis.com/v1beta/openai/"
    return None


def _get_model(provider: str):
    if provider == "groq":
        return GROQ_MODEL
    elif provider == "gemini":
        return GEMINI_MODEL
    return OPENAI_MODEL


def _get_client_for_key(provider: str, api_key: str):
    from openai import OpenAI
    base_url = _get_base_url(provider)
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


def get_openai_client():
    keys = _get_keys(LLM_PROVIDER)
    if not keys:
        raise ValueError(f"No API keys for {LLM_PROVIDER}. Set GROQ_API_KEY, GEMINI_API_KEY, or OPENAI_API_KEY in .env")
    idx = _key_indices[LLM_PROVIDER]
    return _get_client_for_key(LLM_PROVIDER, keys[idx])


def get_model_name():
    return _get_model(LLM_PROVIDER)


def llm_completion(messages, temperature=0.0, max_tokens=500, response_format=None):
    """LLM completion with automatic key rotation on rate limits."""
    provider = LLM_PROVIDER
    keys = _get_keys(provider)
    if not keys:
        raise ValueError(f"No API keys for {provider}")

    model = _get_model(provider)
    key_idx = _key_indices[provider]

    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format:
        kwargs["response_format"] = response_format

    for key_attempt in range(len(keys)):
        current_key = keys[key_idx]
        client = _get_client_for_key(provider, current_key)

        for attempt in range(MAX_RETRIES):
            try:
                response = client.chat.completions.create(**kwargs)
                return response
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "rate_limit" in error_msg.lower() or "quota" in error_msg.lower():
                    wait_time = RETRY_DELAY * (attempt + 1)
                    print(f"  [Rate limit] Key #{key_idx + 1} attempt {attempt + 1}: waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    break
        
        # Rate limit exhausted for this key, rotate to next
        old_idx = key_idx
        key_idx = (key_idx + 1) % len(keys)
        _key_indices[provider] = key_idx
        if key_idx != old_idx:
            print(f"  [Key rotation] {provider}: switched to key #{key_idx + 1}")

    raise Exception(f"All {provider} API keys exhausted (rate limits or errors)")