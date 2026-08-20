"""
TXL Cloud - Gemini fallback
------------------------------------
Used automatically by txl_cloud.py when Groq's free-tier daily quota is
exhausted (or Groq is otherwise unavailable) and GEMINI_API_KEY is set -
keeps chat working instead of erroring out for the rest of the day. Groq
stays the primary backend (faster, and its own separate free quota) -
this only kicks in when Groq itself fails, and only before any part of
the reply has already streamed back (never mixes partial Groq output
with a Gemini continuation).

Setup: get a free key at https://aistudio.google.com/apikey (no credit
card), set GEMINI_API_KEY in the environment (.env locally, or your
host's env vars).
"""

import base64
import os
import re
from typing import Iterator, List, Optional

from google import genai
from google.genai import types

_DATA_URL_RE = re.compile(r"^data:(image/[\w.+-]+);base64,(.+)$", re.DOTALL)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DEFAULT_MODEL = "gemini-3.6-flash"

_client = None


def is_configured() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY"))


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set.")
        _client = genai.Client(api_key=api_key)
    return _client


def _decode_data_url(data_url: str):
    m = _DATA_URL_RE.match(data_url or "")
    if not m:
        raise ValueError("Invalid image data.")
    mime_type, b64 = m.group(1), m.group(2)
    return mime_type, base64.b64decode(b64)


def stream_reply(
    history: List[dict],
    system_prompt: str,
    model: str = DEFAULT_MODEL,
    image_data_url: Optional[str] = None,
) -> Iterator[str]:
    """
    Same output shape as txl_cloud.stream_reply (yields text chunks), but
    takes an already-assembled system_prompt string directly - the caller
    has already merged in custom_instructions etc. If image_data_url is
    set (a "data:image/...;base64,..." string), it's attached to the most
    recent user turn - Gemini is multimodal, unlike the Groq/Ollama models.
    """
    contents = []
    for m in history:
        if not m.get("content"):
            continue
        role = "model" if m["role"] == "assistant" else "user"
        contents.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))

    if image_data_url and contents and contents[-1].role == "user":
        mime_type, image_bytes = _decode_data_url(image_data_url)
        contents[-1].parts.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))

    config = types.GenerateContentConfig(system_instruction=system_prompt, temperature=0.6)
    stream = _get_client().models.generate_content_stream(model=model, contents=contents, config=config)
    for chunk in stream:
        if chunk.text:
            yield chunk.text
