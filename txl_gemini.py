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
IMAGE_MODEL = os.environ.get("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image")

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


def generate_image(prompt: str, model: str = IMAGE_MODEL) -> str:
    """
    Generates a single image from a text prompt via Gemini's native image
    output (not a separate Imagen call - same Content API, just asks for
    an IMAGE part back). Returns a "data:image/...;base64,..." URL, ready
    to drop straight into an <img> tag or store in Message.image_data.
    Raises RuntimeError with a plain-language message on failure (e.g. the
    model refused the prompt, or returned no image part).

    Note: IMAGE_MODEL may need updating over time as Google renames/retires
    image-generation model ids - override with GEMINI_IMAGE_MODEL if the
    default stops working (see README).
    """
    config = types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"])
    try:
        response = _get_client().models.generate_content(model=model, contents=prompt, config=config)
    except Exception as e:
        if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
            raise RuntimeError(
                "Image generation is rate-limited or out of free quota on this Gemini API key right "
                "now (it has its own separate quota from regular chat) - wait a bit and try again, or "
                "check your plan at https://ai.dev/rate-limit."
            ) from e
        raise RuntimeError(f"Image generation failed: {e}") from e

    if not response.candidates:
        raise RuntimeError("The model didn't return anything for that prompt - try rephrasing it.")
    parts = response.candidates[0].content.parts or []
    for part in parts:
        if getattr(part, "inline_data", None) and part.inline_data.data:
            mime_type = part.inline_data.mime_type or "image/png"
            b64 = base64.b64encode(part.inline_data.data).decode("ascii")
            return f"data:{mime_type};base64,{b64}"

    text_parts = [p.text for p in parts if getattr(p, "text", None)]
    if text_parts:
        raise RuntimeError("The model responded with text instead of an image: " + " ".join(text_parts)[:300])
    raise RuntimeError("The model didn't return an image for that prompt - try rephrasing it.")


_PROMPT_EXPANSION_INSTRUCTION = """You are an expert image-prompt engineer. \
Given a short user request, think through composition, subject detail, \
lighting, color palette, mood, and art style, then output ONLY the improved, \
detailed image-generation prompt - no preamble, no explanation, no quotes \
around it, just the final prompt text itself, at most 4 sentences."""


def expand_image_prompt(prompt: str, model: str = DEFAULT_MODEL) -> str:
    """
    'Thinking' mode for image generation: asks the text model to reason
    about and expand a short prompt into a richer one before actually
    generating the image. Falls back to the original prompt unchanged if
    this step fails for any reason - a worse prompt beats no image at all.
    """
    try:
        config = types.GenerateContentConfig(system_instruction=_PROMPT_EXPANSION_INSTRUCTION, temperature=0.8)
        response = _get_client().models.generate_content(model=model, contents=prompt, config=config)
        expanded = (response.text or "").strip()
        return expanded or prompt
    except Exception:
        return prompt
