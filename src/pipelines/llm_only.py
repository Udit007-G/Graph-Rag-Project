from typing import Dict
from config.settings import get_openai_client, get_model_name, llm_completion


def run_llm_only(question: str, tg_client=None) -> Dict:
    """LLM-Only baseline: no retrieval, just the model reasoning."""

    prompt = f"""Answer this question to the best of your ability.
If you don't know the answer, say so.

Question: {question}

Answer:"""

    try:
        response = llm_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )
        answer = response.choices[0].message.content.strip()
        tokens = response.usage.prompt_tokens + response.usage.completion_tokens
    except Exception as e:
        answer = f"Error: {e}"
        tokens = 0

    return {
        "pipeline": "LLM-Only",
        "question": question,
        "answer": answer,
        "total_tokens": tokens
    }
