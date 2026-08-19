"""
YouTube Video Summarizer - Free / Cloud Edition
-------------------------------------------------
$0 cost: uses Groq's free API (fast hosted inference, generous free tier,
no credit card) instead of a local model, so it doesn't touch your PC's
CPU/RAM.

Give it a YouTube URL. It:
  1. Fetches the video's title/channel (via YouTube's public oEmbed
     endpoint - no API key needed).
  2. Fetches the video's transcript/captions (via youtube-transcript-api -
     no API key, no download of the actual video).
  3. Uses Groq to write a short "About this video" blurb plus a detailed,
     topic-by-topic explanation of what's actually covered (not just a
     list of topic names).

Long transcripts are summarized in chunks and then combined (map-reduce),
so this works on long videos too, not just short ones.

Note: unlike a fully-local setup, the transcript text is sent to Groq's
servers to be processed (not 100% private) - that's the trade-off for not
loading down your own machine.

Usage:
    python youtube_agent.py "https://www.youtube.com/watch?v=..."
"""

import argparse
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests
from groq import Groq, RateLimitError
from youtube_transcript_api import YouTubeTranscriptApi

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DEFAULT_MODEL = "openai/gpt-oss-120b"
CHUNK_CHARS = 4_000  # transcript chars per map-reduce chunk (kept small - free tier has a low tokens/minute cap)
COMBINED_NOTES_CHARS = 6_000  # cap on notes fed into the final synthesis pass
CHUNK_WORKERS = 2  # transcript chunks summarized in parallel
MAX_RETRIES = 5

CHUNK_SUMMARY_PROMPT = """This is one part of a video transcript (there may \
be other parts before/after this one). Write detailed notes on what is \
ACTUALLY SAID in this part - specific facts, steps, settings, numbers, \
examples, or reasoning mentioned. Do not just name the topics; capture the \
substance. Use short bullet points. No intro, no "in this part" preamble - \
just the bullets. If this part is just filler/small talk with no real \
content, write "No substantive content in this part."

--- TRANSCRIPT PART ---
{chunk}
"""

FINAL_SUMMARY_PROMPT = """You are writing a summary of a YouTube video \
titled "{title}" by {author}.

Below are detailed notes extracted from the full transcript, in order. \
Using ONLY the information in these notes, write:

**About this video:** 1-2 sentences on what the video is / who it's for.

Then identify the 4-8 main topics discussed, in the order they come up, \
and for EACH one write a section in this exact format:

### <short topic name>
<2-4 sentences explaining what was actually said about this topic - the \
specific steps, settings, examples, comparisons, or reasoning from the \
notes. Do not just restate the topic name - explain the actual content.>

Rules:
- Use ONLY information that appears in the notes below. Do not add outside \
knowledge, do not guess, do not embellish.
- If the notes don't have enough detail to explain a topic properly, say \
"the video mentions this briefly but doesn't go into detail" instead of \
making something up.

Output only the "About this video" line followed by the topic sections - \
nothing else, no closing remarks.

--- NOTES FROM TRANSCRIPT ---
{notes}
"""

SHORT_SUMMARY_PROMPT = """You are writing a summary of a YouTube video \
titled "{title}" by {author}. Below is its full transcript.

Using ONLY what's actually said in the transcript, write:

**About this video:** 1-2 sentences on what the video is / who it's for.

Then identify the main topics discussed (as many as the video actually \
covers - could be just one for a short video), and for EACH one write a \
section in this exact format:

### <short topic name>
<2-4 sentences explaining what was actually said about this topic - \
specific details, not just the topic name.>

Do not add information that isn't in the transcript. Do not guess or \
embellish. Output only the "About this video" line followed by the topic \
sections - nothing else.

--- TRANSCRIPT ---
{transcript}
"""

VERIFY_PROMPT = """You are a fact-checker. Below is a written video \
summary and the exact source material (transcript notes) it's supposed to \
be based on.

Check EVERY factual claim in the summary against the source material. For \
each claim:
- If it's directly supported by the source material, keep it as-is.
- If it is NOT supported (invented, exaggerated, or from outside \
knowledge about the topic rather than what this specific video actually \
said), remove that specific claim or reword the sentence to only state \
what the source material actually says. Do not add any new information.

Output ONLY the corrected summary in the exact same format ("About this \
video:" line followed by ### topic sections) - no commentary about what \
you changed, no preamble.

--- SUMMARY TO CHECK ---
{summary}

--- SOURCE MATERIAL ---
{source}
"""

VIDEO_ID_PATTERNS = [
    r"(?:v=|/)([0-9A-Za-z_-]{11})(?:[&?/]|$)",
    r"youtu\.be/([0-9A-Za-z_-]{11})",
]

_client = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Get a free key (no credit card) at "
                "https://console.groq.com/keys, then either set it as an "
                "environment variable or put GROQ_API_KEY=gsk_... in a .env "
                "file next to this script."
            )
        _client = Groq(api_key=api_key)
    return _client


def extract_video_id(url: str) -> str:
    for pattern in VIDEO_ID_PATTERNS:
        m = re.search(pattern, url)
        if m:
            return m.group(1)
    raise ValueError(f"Couldn't find a YouTube video ID in: {url}")


def get_metadata(url: str) -> dict:
    resp = requests.get(
        "https://www.youtube.com/oembed",
        params={"url": url, "format": "json"},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    return {
        "title": data.get("title", "Unknown title"),
        "author": data.get("author_name", "Unknown channel"),
        "thumbnail": data.get("thumbnail_url"),
    }


def get_transcript(video_id: str) -> str:
    ytt_api = YouTubeTranscriptApi()
    fetched = ytt_api.fetch(video_id)
    return " ".join(snippet.text for snippet in fetched)


def summarize(transcript: str, title: str, author: str, model: str) -> str:
    if len(transcript) <= CHUNK_CHARS:
        prompt = SHORT_SUMMARY_PROMPT.format(title=title, author=author, transcript=transcript)
        summary = _ask(model, prompt)
        source = transcript
    else:
        chunks = [transcript[i:i + CHUNK_CHARS] for i in range(0, len(transcript), CHUNK_CHARS)]
        print(f"  [summarizing {len(chunks)} parts in parallel]", file=sys.stderr)
        with ThreadPoolExecutor(max_workers=CHUNK_WORKERS) as pool:
            notes = list(pool.map(
                lambda c: _ask(model, CHUNK_SUMMARY_PROMPT.format(chunk=c)), chunks
            ))

        combined_notes = "\n\n".join(notes)[:COMBINED_NOTES_CHARS]
        print("  [writing final summary]", file=sys.stderr)
        prompt = FINAL_SUMMARY_PROMPT.format(title=title, author=author, notes=combined_notes)
        summary = _ask(model, prompt)
        source = combined_notes

    print("  [verifying accuracy]", file=sys.stderr)
    verify_prompt = VERIFY_PROMPT.format(summary=summary, source=source)
    return _ask(model, verify_prompt)


def _wait_seconds_from_error(e: Exception) -> float:
    m = re.search(r"try again in ([\d.]+)s", str(e))
    return float(m.group(1)) + 0.5 if m else 3.0


def _ask(model: str, prompt: str) -> str:
    for attempt in range(MAX_RETRIES):
        try:
            response = _get_client().chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            return (response.choices[0].message.content or "").strip()
        except RateLimitError as e:
            if attempt == MAX_RETRIES - 1:
                raise
            wait = _wait_seconds_from_error(e)
            print(f"    [rate limited, waiting {wait:.1f}s]", file=sys.stderr)
            time.sleep(wait)


def run(url: str, model: str = DEFAULT_MODEL) -> str:
    video_id = extract_video_id(url)
    print(f"Fetching video info for {video_id} ...", file=sys.stderr)
    meta = get_metadata(url)

    print("Fetching transcript ...", file=sys.stderr)
    try:
        transcript = get_transcript(video_id)
    except Exception as e:
        raise RuntimeError(
            f"Couldn't get a transcript for this video ({e}). "
            "It may have captions disabled."
        )

    if not transcript.strip():
        raise RuntimeError("Transcript came back empty - this video may have no captions.")

    print(f"Summarizing ({len(transcript):,} characters of transcript) ...", file=sys.stderr)
    summary = summarize(transcript, meta["title"], meta["author"], model)

    thumbnail_md = f"![{meta['title']}]({meta['thumbnail']})\n\n" if meta.get("thumbnail") else ""

    return (
        f"# {meta['title']}\n\n"
        f"{thumbnail_md}"
        f"**Channel:** {meta['author']}  \n"
        f"**URL:** {url}\n\n"
        f"{summary}\n"
    )


def main():
    parser = argparse.ArgumentParser(description="Summarize a YouTube video using Groq's free API + its transcript.")
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Groq model to use (default: {DEFAULT_MODEL})")
    args = parser.parse_args()

    try:
        report = run(args.url, args.model)
    except Exception as e:
        sys.exit(str(e))

    print(report)


if __name__ == "__main__":
    main()
