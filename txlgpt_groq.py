"""
Txl GPT - Groq backend
------------------------------------
Txl GPT's own Groq connector - deliberately a separate file from TXL
Cloud's (txl_cloud.py), even though the underlying approach is similar,
so the two apps share no code, no persona, and no runtime state. Txl GPT
is its own product.

Setup: get a free Groq API key (no credit card) at
https://console.groq.com/keys, then set GROQ_API_KEY (or the
Txl-GPT-specific TXLGPT_GROQ_API_KEY, for its own separate quota) in a
.env file - see .env.example.
"""

import json
import os
import re
import sys
import time
from typing import Iterator, List, Tuple

from groq import Groq, RateLimitError

import txlgpt_gemini as vision

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DEFAULT_MODEL = "openai/gpt-oss-120b"
MAX_RETRIES = 5
MAX_HISTORY_MESSAGES = 24  # ~12 user/assistant turns kept; oldest trimmed first

SYSTEM_PROMPT = """You are Txl GPT, a free, helpful AI chat assistant. \
You are your own independent product, built to run on a free open-weight \
model rather than OpenAI's ChatGPT or Anthropic's Claude - if asked, be \
upfront about that rather than claiming to be either of them. Be clear and \
warm. Prioritize being accurate and substantive over sounding confident - \
give real, specific, correct answers with the actual details/numbers/names \
requested; if you're not sure of something, say so plainly rather than \
guessing or filling space with vague, generic-sounding filler. Don't hedge \
on things you do know just to seem cautious.

Match your depth to the request. A quick factual question ("what's 12% of \
80", "what year did X happen") gets a short, direct answer - don't pad it \
with headers or bullet after-thoughts it doesn't need. But when asked to \
explain, analyze, describe, or walk through something substantial - an \
image, a document, a comparison, a concept - go deep: don't just transcribe \
or list what's there, explain what each part actually means and why it \
matters, the way a knowledgeable person would walk a colleague through it. \
Use headings and bullets to structure a longer explanation, and draw on \
what you actually know about the subject (not just what's visible) to add \
real context per point - a name alone ("Feature X") is worth far less than \
a sentence on what Feature X does and why someone would care. Thin, \
list-only answers to substantial questions are a failure mode - elaborate. \
Use markdown (headings, lists, code blocks) when it genuinely helps \
readability. You're also a capable coding assistant: when asked for code, \
write correct, working code and always put it in a fenced code block \
tagged with the right language (e.g. ```python) so it can be \
syntax-highlighted - explain briefly around it, but don't pad with \
unnecessary commentary."""

_client = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        # TXLGPT_GROQ_API_KEY, if set, gives Txl GPT its own Groq
        # account/quota, independent from every other agent in this repo.
        # Falls back to the shared GROQ_API_KEY if not set.
        api_key = os.environ.get("TXLGPT_GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
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


def stream_reply(
    history: List[dict],
    model: str = DEFAULT_MODEL,
    custom_instructions: str = None,
    image_data_url: str = None,
    system_prompt: str = None,
    reasoning_effort: str = None,
) -> Iterator[str]:
    """
    Given a conversation history (list of {"role", "content"} dicts, no
    system message - just user/assistant turns), stream the assistant's
    reply back as text chunks. `custom_instructions`, if set, is the
    user's own standing preferences, appended to the system prompt.
    `system_prompt`, if set, replaces the default Txl GPT persona entirely
    (used when a chat is running under a custom GPT). `reasoning_effort`
    ("low"/"medium"/"high"), if set, is passed straight through to Groq -
    only openai/gpt-oss-* models honor it (they're native reasoning
    models).
    """
    system_prompt = system_prompt or SYSTEM_PROMPT
    if custom_instructions:
        system_prompt += (
            "\n\nThe user has also given you these standing preferences for how "
            "they'd like you to behave - follow them unless they conflict with "
            f"being safe or honest:\n{custom_instructions}"
        )
    trimmed = trim_history(history)

    # None of Groq's current free models are multimodal - a pasted image
    # can only be understood by Gemini, so route straight there and skip
    # Groq entirely for this turn.
    if image_data_url:
        if not vision.is_configured():
            raise RuntimeError(
                "Image understanding isn't set up on this deployment yet - "
                "ask whoever runs it to configure GEMINI_API_KEY."
            )
        yield from vision.stream_reply(trimmed, system_prompt, image_data_url=image_data_url)
        return

    messages = [{"role": "system", "content": system_prompt}] + trimmed

    last_error = None
    yielded_any = False
    effort = reasoning_effort
    for attempt in range(MAX_RETRIES):
        try:
            kwargs = {"reasoning_effort": effort} if effort else {}
            stream = _get_client().chat.completions.create(
                model=model, messages=messages, temperature=0.6, stream=True, **kwargs,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yielded_any = True
                    yield delta
            return
        except RateLimitError as e:
            last_error = e
            wait = _wait_seconds_from_error(e)
            # A short wait is a per-minute limit - worth retrying. A long
            # one (or the daily-cap message) means Groq won't recover
            # within this request, so stop retrying and fall back instead.
            if wait <= 30 and attempt < MAX_RETRIES - 1:
                print(f"    [rate limited, waiting {wait:.1f}s]", file=sys.stderr)
                time.sleep(wait)
                continue
            break
        except Exception as e:
            last_error = e
            break

    # Groq didn't come through. Fall back to Gemini if it's configured -
    # but only if nothing has streamed back yet, so a reply is never a
    # broken mix of partial Groq output plus a Gemini continuation.
    if not yielded_any and vision.is_configured():
        try:
            print(f"    [Groq unavailable ({last_error}) - falling back to Gemini]", file=sys.stderr)
            yield from vision.stream_reply(trimmed, system_prompt)
            return
        except Exception as gemini_error:
            raise RuntimeError(
                f"Groq failed ({last_error}), and the Gemini fallback also failed ({gemini_error})."
            ) from gemini_error
    if last_error:
        raise last_error


def _groq_ready_messages(messages: List[dict]) -> List[dict]:
    """Groq/OpenAI-style APIs need tool_calls[].function.arguments as a JSON
    string; our shared in-memory history keeps it as a plain dict, so it
    also works unchanged for Ollama, which wants a dict there instead.
    Convert only right before sending."""
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
    Work mode, not regular chat). `messages` should already include the
    system prompt. Returns (text, tool_calls):
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
        sys.exit('Usage: python txlgpt_groq.py "your message"')
    message = " ".join(sys.argv[1:])
    for chunk in stream_reply([{"role": "user", "content": message}]):
        print(chunk, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
