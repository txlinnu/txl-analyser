"""
TXL Claude - Free Chat Agent
------------------------------------
A Claude-style chat assistant, powered by Groq's free API (open-weight
models, generous free tier, no credit card) instead of a paid model.

Conversation history lives only in memory, keyed to a browser session
cookie - never written to disk. It's not 100% private (your messages are
sent to Groq's servers for inference, same trade-off as the other agents
in this repo), but nothing is logged, stored in a database, or tied to an
account - because there isn't one.

Setup: same GROQ_API_KEY as the other agents here (see .env.example).
"""

import json
import os
import re
import sys
import time
from typing import Iterator, List, Tuple

from groq import Groq, RateLimitError

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DEFAULT_MODEL = "openai/gpt-oss-120b"
MAX_RETRIES = 5
MAX_HISTORY_MESSAGES = 24  # ~12 user/assistant turns kept; oldest trimmed first

SYSTEM_PROMPT = """You are TXL Claude, a free, private, helpful AI chat \
assistant. You're inspired by Claude's helpful/honest/concise style, but \
you are a separate, independent assistant, built to run on a free \
open-weight model (via Groq) rather than Anthropic's Claude models - if \
asked, be upfront about that rather than claiming to be Claude itself. \
Be clear, warm, and concise. Say when you don't know something instead \
of guessing. Use markdown (headings, lists, code blocks) when it \
genuinely helps readability, but don't over-format short answers. You're \
also a capable coding assistant: when asked for code, write correct, \
working code and always put it in a fenced code block tagged with the \
right language (e.g. ```python) so it can be syntax-highlighted - explain \
briefly around it, but don't pad with unnecessary commentary."""

_client = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        # CHAT_GROQ_API_KEY, if set, lets TXL Claude use a different Groq
        # account/quota than the other agents here (pdf_research_agent_local.py,
        # youtube_agent.py) - useful since they'd otherwise all share one
        # daily free-tier token limit. Falls back to the same GROQ_API_KEY
        # everything else uses if it's not set.
        api_key = os.environ.get("CHAT_GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Get a free key (no credit card) at "
                "https://console.groq.com/keys, then put it in a .env file "
                "as GROQ_API_KEY=gsk_..."
            )
        _client = Groq(api_key=api_key)
    return _client


def _wait_seconds_from_error(e: Exception) -> float:
    m = re.search(r"try again in ([\d.]+)s", str(e))
    return float(m.group(1)) + 0.5 if m else 3.0


def trim_history(history: List[dict]) -> List[dict]:
    if len(history) > MAX_HISTORY_MESSAGES:
        return history[-MAX_HISTORY_MESSAGES:]
    return history


def stream_reply(history: List[dict], model: str = DEFAULT_MODEL, custom_instructions: str = None) -> Iterator[str]:
    """
    Given a conversation history (list of {"role", "content"} dicts, no
    system message - just user/assistant turns), stream the assistant's
    reply back as text chunks. `custom_instructions`, if set, is the
    user's own "Customize" preferences, appended to the system prompt.
    """
    system_prompt = SYSTEM_PROMPT
    if custom_instructions:
        system_prompt += (
            "\n\nThe user has also given you these standing preferences for how "
            "they'd like you to behave - follow them unless they conflict with "
            f"being safe or honest:\n{custom_instructions}"
        )
    messages = [{"role": "system", "content": system_prompt}] + trim_history(history)

    for attempt in range(MAX_RETRIES):
        try:
            stream = _get_client().chat.completions.create(
                model=model, messages=messages, temperature=0.6, stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
            return
        except RateLimitError as e:
            if attempt == MAX_RETRIES - 1:
                raise
            wait = _wait_seconds_from_error(e)
            print(f"    [rate limited, waiting {wait:.1f}s]", file=sys.stderr)
            time.sleep(wait)


def _groq_ready_messages(messages: List[dict]) -> List[dict]:
    """Groq/OpenAI-style APIs need tool_calls[].function.arguments as a JSON
    string; our shared in-memory history keeps it as a plain dict (see
    chat_app.py's _tool_call_message) so it also works unchanged for Ollama,
    which wants a dict there instead. Convert only right before sending."""
    prepared = []
    for m in messages:
        if m.get("tool_calls"):
            m = dict(m)
            m["tool_calls"] = [
                {**tc, "function": {**tc["function"], "arguments": json.dumps(tc["function"]["arguments"])}}
                if isinstance(tc["function"].get("arguments"), dict) else tc
                for tc in m["tool_calls"]
            ]
        prepared.append(m)
    return prepared


def run_with_tools(messages: List[dict], tools: List[dict], model: str = DEFAULT_MODEL) -> Tuple[str, List[dict]]:
    """
    One non-streaming turn with function-calling tools available (used by
    the Code-mode agent, not the regular chat). `messages` should already
    include the system prompt. Returns (text, tool_calls):
      - if the model wants to call tools: (None, [{"id","name","arguments"}, ...])
      - if the model answered in text: (text, [])
    """
    messages = _groq_ready_messages(messages)
    for attempt in range(MAX_RETRIES):
        try:
            response = _get_client().chat.completions.create(
                model=model, messages=messages, tools=tools, tool_choice="auto", temperature=0.3,
            )
            break
        except RateLimitError as e:
            if attempt == MAX_RETRIES - 1:
                raise
            wait = _wait_seconds_from_error(e)
            print(f"    [rate limited, waiting {wait:.1f}s]", file=sys.stderr)
            time.sleep(wait)

    msg = response.choices[0].message
    if msg.tool_calls:
        calls = []
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append({"id": tc.id, "name": tc.function.name, "arguments": args})
        return None, calls
    return (msg.content or "").strip(), []


def main():
    """Quick CLI smoke test: a single-turn chat from argv."""
    if len(sys.argv) < 2:
        sys.exit('Usage: python txl_claude.py "your message"')
    message = " ".join(sys.argv[1:])
    for chunk in stream_reply([{"role": "user", "content": message}]):
        print(chunk, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
