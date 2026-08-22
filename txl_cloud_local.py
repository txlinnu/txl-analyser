"""
TXL Cloud - Fully Local Edition
------------------------------------
$0 cost AND fully private AND unlimited: runs entirely on your own PC via
Ollama - no API key, no daily token cap, nothing ever sent anywhere. The
trade-off vs the Groq edition (txl_cloud.py) is speed and setup: replies
are only as fast as your hardware, and you need to install Ollama and
download a model (several GB) first.

Setup (one-time), on the machine that will run this:
    1. Install Ollama: https://ollama.com/download
    2. Pull the models this app expects:
         ollama pull qwen2.5:14b     (Accurate - bigger, slower - needs ~9 GB free)
         ollama pull qwen2.5:7b      (Balanced - default - ~4.7 GB)
         ollama pull qwen2.5vl:7b    (optional - needed to understand pasted images, ~6 GB)
    3. pip install ollama

Then run chat_app.py with CHAT_BACKEND=ollama set (see README.md).

Sizing note: tuned for a machine with 32GB RAM + a small (~4GB) GPU -
Ollama automatically offloads what fits onto the GPU and runs the rest on
CPU. Bigger models are noticeably slower than Groq's cloud replies (rough
ballpark: a few tokens/second on that kind of hardware, vs near-instant)
- that's the cost of "unlimited and private" over "fast". If qwen2.5:14b
feels too slow, use qwen2.5:7b (the default) or an even smaller model of
your choice (pull it, then add it to MODEL_CHOICES in chat_app.py).
"""

import json
import sys
from typing import Iterator, List, Tuple

import ollama

DEFAULT_MODEL = "qwen2.5:7b"
MAX_HISTORY_MESSAGES = 24  # ~12 user/assistant turns kept; oldest trimmed first

# qwen2.5:7b/14b are text-only - a pasted image needs a vision-capable
# model instead. Pull it separately: ollama pull qwen2.5vl:7b (~6 GB)
VISION_MODEL = "qwen2.5vl:7b"

SYSTEM_PROMPT = """You are TXL Cloud, a free, private, helpful AI chat \
assistant. You're inspired by Claude's helpful/honest/concise style, but \
you are a separate, independent assistant, running entirely locally on \
this machine via an open-weight model - if asked, be upfront about that \
rather than claiming to be Claude itself. Be clear, warm, and concise. \
Prioritize being accurate and substantive over sounding confident - give \
real, specific, correct answers with the actual details/numbers/names \
requested; if you're not sure of something, say so plainly rather than \
guessing or filling space with vague, generic-sounding filler. Don't \
hedge on things you do know just to seem cautious. Use markdown \
(headings, lists, code blocks) when it genuinely helps readability, but \
don't over-format short answers. You're also a capable coding assistant: \
when asked for code, write correct, working code and always put it in a \
fenced code block tagged with the right language (e.g. ```python) so it \
can be syntax-highlighted - explain briefly around it, but don't pad \
with unnecessary commentary. Fenced ```html or ```svg code blocks get an \
actual live rendered preview shown to the user (not just syntax-highlighted \
text) - so when a webpage, UI mockup, game, chart, or diagram would \
genuinely help, write one as a single self-contained ```html block with \
any CSS/JS inlined (no external file references, since nothing else can \
be loaded), or a ```svg block for a static graphic. Only do this when \
it's actually the right way to answer - don't force an HTML block onto a \
request that's better served by a normal text or code answer."""


def trim_history(history: List[dict]) -> List[dict]:
    if len(history) > MAX_HISTORY_MESSAGES:
        return history[-MAX_HISTORY_MESSAGES:]
    return history


def _data_url_to_b64(data_url: str) -> str:
    return data_url.split(",", 1)[1] if "," in data_url else data_url


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
    reply back as text chunks. Talks to a local Ollama server
    (http://127.0.0.1:11434 by default). `custom_instructions`, if set, is
    the user's own "Customize" preferences, appended to the system prompt.
    If image_data_url is set, the request is routed to VISION_MODEL
    instead - qwen2.5:7b/14b can't see images, only qwen2.5vl can.
    `system_prompt`, if set, replaces the default TXL Cloud persona
    entirely - lets other apps (e.g. txlgpt_app.py) reuse this connector
    with their own identity instead of TXL Cloud's. `reasoning_effort` is
    accepted for call-signature parity with txl_cloud.py but ignored here -
    qwen2.5 isn't a native reasoning model, unlike Groq's gpt-oss.
    """
    system_prompt = system_prompt or SYSTEM_PROMPT
    if custom_instructions:
        system_prompt += (
            "\n\nThe user has also given you these standing preferences for how "
            "they'd like you to behave - follow them unless they conflict with "
            f"being safe or honest:\n{custom_instructions}"
        )
    messages = [{"role": "system", "content": system_prompt}] + trim_history(history)

    if image_data_url:
        model = VISION_MODEL
        if messages and messages[-1]["role"] == "user":
            messages[-1] = {**messages[-1], "images": [_data_url_to_b64(image_data_url)]}

    try:
        stream = ollama.chat(model=model, messages=messages, stream=True)
        for chunk in stream:
            content = chunk["message"]["content"]
            if content:
                yield content
    except ollama.ResponseError as e:
        if e.status_code == 404:
            raise RuntimeError(
                f"Model '{model}' isn't pulled yet. Run: ollama pull {model}"
            ) from e
        raise RuntimeError(str(e)) from e
    except Exception as e:
        raise RuntimeError(
            "Can't reach Ollama on this machine. Make sure it's installed and "
            "running - it usually starts automatically after install; "
            "otherwise run `ollama serve` in a terminal and try again. "
            f"(Details: {e})"
        ) from e


def _ollama_ready_messages(messages: List[dict]) -> List[dict]:
    """Ollama's client validates tool_calls[].function.arguments as a dict
    (unlike Groq/OpenAI, which want a JSON string there) - our shared
    in-memory history keeps it as a dict already (see chat_app.py's
    _tool_call_message), so this just guards against a stray string."""
    prepared = []
    for m in messages:
        if m.get("tool_calls"):
            m = dict(m)
            new_calls = []
            for tc in m["tool_calls"]:
                args = tc["function"].get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    tc = {**tc, "function": {**tc["function"], "arguments": args}}
                new_calls.append(tc)
            m["tool_calls"] = new_calls
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
    messages = _ollama_ready_messages(messages)
    try:
        response = ollama.chat(model=model, messages=messages, tools=tools)
    except ollama.ResponseError as e:
        if e.status_code == 404:
            raise RuntimeError(f"Model '{model}' isn't pulled yet. Run: ollama pull {model}") from e
        raise RuntimeError(str(e)) from e
    except Exception as e:
        raise RuntimeError(
            "Can't reach Ollama on this machine. Make sure it's installed and "
            f"running - otherwise run `ollama serve` and try again. (Details: {e})"
        ) from e

    msg = response["message"]
    raw_calls = msg.get("tool_calls") or []
    if raw_calls:
        calls = []
        for i, tc in enumerate(raw_calls):
            fn = tc["function"]
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            calls.append({"id": tc.get("id") or f"call_{i}", "name": fn["name"], "arguments": args})
        return None, calls
    return (msg.get("content") or "").strip(), []


def main():
    """Quick CLI smoke test: a single-turn chat from argv."""
    if len(sys.argv) < 2:
        sys.exit('Usage: python txl_cloud_local.py "your message"')
    message = " ".join(sys.argv[1:])
    for chunk in stream_reply([{"role": "user", "content": message}]):
        print(chunk, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
