from __future__ import annotations
import os

LLM_TEMPLATE = '''You are an expert code reviewer. Given raw linter outputs below,
produce a concise review with:
1) High-level summary (what's good / what needs work)
2) Prioritized issues (bullet points)
3) Concrete suggestions (diff-like hints if possible)
Be precise but encouraging; keep it under ~300 words.

<ruff>
{ruff}
</ruff>

<pylint>
{pylint}
</pylint>
'''

def llm_available() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))

def generate_review(ruff_out: str, pylint_out: str) -> str:
    """
    If OPENAI_API_KEY is present, attempt to use OpenAI's SDK; otherwise return an empty string.
    Users can replace with their provider of choice.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return ""

    try:
        # Lazy import to avoid mandatory dependency
        from openai import OpenAI  # openai>=1.0.0
        client = OpenAI(api_key=api_key)
        prompt = LLM_TEMPLATE.format(ruff=ruff_out[:8000], pylint=pylint_out[:8000])

        chat = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a senior engineer performing code review."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return chat.choices[0].message.content or ""
    except Exception as e:
        return f"_LLM review unavailable: {e}_"
