"""Shared config and the LLM client wrapper.

Defaults to Groq (free tier, plain REST, no SDK/grpc issues). Gemini, Anthropic
and OpenAI are also supported -- switch LLM_PROVIDER in .env.
"""
import os
import sys
import time
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
KB_DIR = os.path.join(ROOT, "kb")
PRACTICE_DIR = os.path.join(ROOT, "practice")

# Dataset is frozen on this date (see kb/00_about_this_knowledge_base.md [KB-00.2])
TODAY = date(2026, 10, 20)

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "groq")  # groq | gemini | anthropic | openai
_DEFAULT_MODELS = {
    "groq": "qwen/qwen3.8-27b",
    "gemini": "gemini-3.5-flash",
    "anthropic": "claude-sonnet-4-5",
    "openai": "gpt-4o-mini",
}
LLM_MODEL = os.environ.get("LLM_MODEL", _DEFAULT_MODELS.get(LLM_PROVIDER, ""))

SYSTEM_PROMPT_BASE = """You are ZentroMart's AI support assistant. ZentroMart is an Indian \
online store selling groceries and electronics. You help customers with orders, \
deliveries, returns, refunds, cancellations and warranty questions.

Rules you must always follow:
- In your very first reply to a customer, say you are an AI assistant.
- Answer ONLY using the information you are given (documents / order data). \
Never invent a policy, date, amount or order detail.
- If the provided information does not cover the question, say you don't know \
and offer to connect the customer to a human colleague.
- Keep replies to two or three short sentences unless the customer asked for detail.
- Reply in the language the customer used.
- Never claim you performed an action (lookup, cancellation, ticket) unless a tool \
result actually confirms it.
"""


def _post_with_retry(url, json_body, headers, provider_name, rpm_wait=15):
    """Shared retry/backoff for any REST provider. 503 = transient overload
    (short backoff); 429 = rate/quota limit (longer backoff, since free tiers
    often need real time to reset, not just a quick retry)."""
    import requests
    last_err = None
    for attempt in range(5):
        try:
            resp = requests.post(url, json=json_body, headers=headers, timeout=60)
        except requests.exceptions.RequestException as e:
            # network-level failure (timeout, reset, DNS blip) -- not an HTTP
            # error code, so it needs its own retry path
            last_err = str(e)
            if attempt < 4:
                wait = 2 ** attempt
                print(f"  [{provider_name} network error, retrying in {wait:.0f}s: {e}]", file=sys.stderr)
                time.sleep(wait)
                continue
            raise RuntimeError(f"{provider_name} kept failing after retries: {last_err}")
        if resp.status_code in (503, 429) and attempt < 4:
            last_err = f"{resp.status_code} {resp.text[:300]}"
            retry_after = resp.headers.get("Retry-After")
            if retry_after:
                wait = float(retry_after)
            elif resp.status_code == 429:
                wait = rpm_wait * (attempt + 1)
            else:
                wait = 2 ** attempt
            print(f"  [{provider_name} {resp.status_code}, retrying in {wait:.0f}s...]", file=sys.stderr)
            time.sleep(wait)
            continue
        if resp.status_code >= 400:
            print(f"  [{provider_name} error body: {resp.text[:500]}]", file=sys.stderr)
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError(f"{provider_name} kept returning errors after retries: {last_err}")


def _groq_generate(system: str, messages: list, max_tokens: int) -> str:
    """Groq's chat completions API is OpenAI-compatible. Plain REST, no SDK."""
    api_key = os.environ.get("GROQ_API_KEY")
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "model": LLM_MODEL,
        "max_tokens": max_tokens,
        "messages": [{"role": "system", "content": system}] + messages,
        "tool_choice": "none",  # some Groq models (gpt-oss family) have a built-in
                                  # browser-search tool that can fire unprompted; this
                                  # is the documented (if imperfectly enforced) way to
                                  # disable it. qwen/qwen3.8-27b, our default, has no
                                  # such built-in tool, so this is a belt-and-suspenders.
    }
    data = _post_with_retry(url, body, headers, "groq")
    return data["choices"][0]["message"]["content"]


def _gemini_generate(system: str, messages: list, max_tokens: int) -> str:
    """Plain REST call to Gemini (no SDK / no grpc) -- avoids the native-DLL
    dependency that the google-generativeai package pulls in via grpc, which
    some locked-down Windows machines (Device Guard / WDAC) block."""
    api_key = os.environ.get("GEMINI_API_KEY")
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{LLM_MODEL}:generateContent?key={api_key}")
    contents = [{"role": "model" if m["role"] == "assistant" else "user",
                 "parts": [{"text": m["content"]}]} for m in messages]
    body = {
        "contents": contents,
        "systemInstruction": {"parts": [{"text": system}]},
        "generationConfig": {"maxOutputTokens": max_tokens},
    }
    headers = {"Api-Revision": "2026-05-20"}  # required by Gemini 3.x models, e.g. gemini-3.5-flash
    data = _post_with_retry(url, body, headers, "gemini")
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        raise RuntimeError(f"Unexpected Gemini response: {data}")


def call_llm(system: str, user: str, max_tokens: int = 600) -> str:
    """Single-turn completion. Returns the assistant's text."""
    return call_llm_chat(system, [{"role": "user", "content": user}], max_tokens)


def call_llm_chat(system: str, messages: list, max_tokens: int = 600) -> str:
    """Multi-turn completion. messages = [{"role": "user"/"assistant", "content": str}, ...]"""
    if LLM_PROVIDER == "groq":
        return _groq_generate(system, messages, max_tokens)
    elif LLM_PROVIDER == "gemini":
        return _gemini_generate(system, messages, max_tokens)
    elif LLM_PROVIDER == "anthropic":
        import anthropic
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=LLM_MODEL, max_tokens=max_tokens, system=system, messages=messages,
        )
        return "".join(b.text for b in resp.content if b.type == "text")
    elif LLM_PROVIDER == "openai":
        from openai import OpenAI
        client = OpenAI()
        resp = client.chat.completions.create(
            model=LLM_MODEL, max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}] + messages,
        )
        return resp.choices[0].message.content
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")
