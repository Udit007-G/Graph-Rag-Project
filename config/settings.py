import os
import time
import random
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
RETRY_DELAY = 10

# Track key usage for rotation
_key_indices = {"groq": 0, "gemini": 0, "openai": 0}
_key_health = {"groq": {}, "gemini": {}, "openai": {}}


def _get_keys(provider: str):
    """Get list of API keys for provider."""
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
    """Create client for specific API key."""
    from openai import OpenAI
    base_url = _get_base_url(provider)
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


def _rotate_key(provider: str):
    """Rotate to next available key for provider."""
    keys = _get_keys(provider)
    if len(keys) <= 1:
        return None
    idx = _key_indices[provider]
    # Try next key
    for i in range(1, len(keys)):
        next_idx = (idx + i) % len(keys)
        if _key_health[provider].get(keys[next_idx], 0) < 3:  # Max 3 failures
            _key_indices[provider] = next_idx
            print(f"  [Key rotation] {provider}: switched to key #{next_idx + 1}")
            return keys[next_idx]
    return None


def _mark_key_failure(provider: str, api_key: str):
    """Track key failures for health checking."""
    _key_health[provider][api_key] = _key_health[provider].get(api_key, 0) + 1


def _mark_key_success(provider: str, api_key: str):
    """Reset failure count on success."""
    if api_key in _key_health[provider]:
        _key_health[provider][api_key] = 0


def get_openai_client():
    """Get OpenAI-compatible client with current key."""
    keys = _get_keys(LLM_PROVIDER)
    if not keys:
        raise ValueError(f"No API keys for {LLM_PROVIDER}. Set GROQ_API_KEY, GEMINI_API_KEY, or OPENAI_API_KEY in .env")
    idx = _key_indices[LLM_PROVIDER]
    return _get_client_for_key(LLM_PROVIDER, keys[idx])


def get_model_name():
    return _get_model(LLM_PROVIDER)


def llm_completion(messages, temperature=0.0, max_tokens=500, response_format=None):
    """LLM completion with key rotation on rate limits."""
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

    # Try each key up to MAX_RETRIES times
    for key_attempt in range(len(keys)):
        current_key = keys[key_idx]
        client = _get_client_for_key(provider, current_key)

        for attempt in range(MAX_RETRIES):
            try:
                response = client.chat.completions.create(**kwargs)
                _mark_key_success(provider, current_key)
                return response
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "rate_limit" in error_msg.lower() or "quota" in error_msg.lower():
                    wait_time = RETRY_DELAY * (attempt + 1)
                    print(f"  [Rate limit] Key #{key_idx + 1} attempt {attempt + 1}: waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    # Non-rate-limit error, don't retry this key
                    break
        
        # This key exhausted, mark failure and rotate
        _mark_key_failure(provider, current_key)
        next_key = _rotate_key(provider)
        if next_key:
            key_idx = _key_indices[provider]
            continue
        break

    raise Exception(f"All {provider} API keys exhausted (rate limits or errors)")
