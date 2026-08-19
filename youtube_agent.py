"""
YouTube Video Summarizer - Free / Cloud Edition
-------------------------------------------------
$0 cost: uses Groq's free API (fast hosted inference, generous free tier,
no credit card) instead of a local model, so it doesn't touch your PC's
CPU/RAM - and free DuckDuckGo search + image search instead of a paid tool.

Give it a YouTube URL. It:
  1. Fetches the video's title/channel (via YouTube's public oEmbed
     endpoint - no API key needed).
  2. Fetches the video's transcript/captions (via youtube-transcript-api -
     no API key, no download of the actual video) - or uses a pasted
     transcript if one is provided.
  3. Condenses the transcript into notes and identifies the main topics.
  4. For EVERY topic, deterministically runs a real DuckDuckGo search and
     fetches the top result's page content, plus one relevant image - the
     same grounding approach used for PDFs, so a video summary isn't just
     restating the transcript, it's enriched with real current context.
  5. Writes a short, sourced, illustrated summary of every topic.

Long transcripts are condensed in chunks first (map-reduce), so this
works on long videos too, not just short ones.

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
from concurrent.futures import ThreadPoolExecutor

import requests
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.proxies import GenericProxyConfig

from pdf_research_agent_local import ask, fetch_page, image_search, web_search

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DEFAULT_MODEL = "openai/gpt-oss-120b"
CHUNK_CHARS = 4_000  # transcript chars per map-reduce chunk (kept small - free tier has a low tokens/minute cap)
COMBINED_NOTES_CHARS = 6_000  # cap on condensed notes used for topic research
CHUNK_WORKERS = 2  # transcript chunks condensed in parallel
MAX_TOPICS = 6
SEARCH_RESULTS_PER_TOPIC = 4
PAGES_TO_FETCH_PER_TOPIC = 1
TOPIC_RESEARCH_WORKERS = 2  # topics researched in parallel (kept low - free tier has a low tokens/minute cap)
CONTEXT_CHARS = 3500  # notes / search results chars per prompt (keeps calls under the free-tier TPM limit)

TRANSCRIPT_ERROR_NOTE = """This video's transcript is auto-generated and \
may contain speech-recognition (ASR) errors - especially garbled technical \
terms, product/brand names, or jargon mistranscribed into a similar- \
sounding but wrong word (e.g. "Jobex" instead of "Zabbix", a monitoring \
tool). When a word clearly doesn't fit the context but a real, well-known \
term that sounds similar does, treat it as the corrected term instead of \
repeating the garbled version. Only do this when you're confident - don't \
guess at unclear non-technical speech."""

CHUNK_SUMMARY_PROMPT = """This is one part of a video transcript (there may \
be other parts before/after this one). Write detailed notes on what is \
ACTUALLY SAID in this part - specific facts, steps, settings, numbers, \
examples, or reasoning mentioned. Do not just name the topics; capture the \
substance. Use short bullet points. No intro, no "in this part" preamble - \
just the bullets. If this part is just filler/small talk with no real \
content, write "No substantive content in this part."

""" + TRANSCRIPT_ERROR_NOTE + """

--- TRANSCRIPT PART ---
{chunk}
"""

TOPIC_EXTRACTION_PROMPT = """You will be given notes on a YouTube video's \
content. List the main topics/subjects it covers, from most to least \
important - at most {max_topics}, but FEWER if the video doesn't support \
that many distinct topics. Do not pad the list or split one idea into \
multiple near-duplicate topics just to reach {max_topics}: a short or \
narrow video might genuinely only have 1-3 real topics, and that's fine. \
Each topic must be clearly distinct from the others.

Output ONLY a numbered list, one short topic name per line (3-6 words \
each), nothing else - no preamble, no explanation.
"""

OVERVIEW_PROMPT = """Write a single short sentence or two (plain language, \
no jargon) summarizing what this video is about overall and who it's for, \
based on the notes below. Output only that, nothing else."""

TOPIC_WRITE_PROMPT = """You are writing one section of a video summary \
about the topic: "{topic}"

Below is (1) notes on what the video says related to this topic, and (2) \
real, current web search results about this topic gathered just now.

Write a short section in this exact format:

### {topic}

<2-5 sentences in plain, jargon-free language that combines what the video \
actually said with useful context the web search adds. If the web results \
add nothing new, just summarize what the video said. If they conflict, \
briefly say so.>

**Sources:**
<a markdown bullet list of the URLs you actually used from the search \
results below - copy them exactly. If nothing from the web was useful, \
write "Video only".>

Do not invent URLs. Only cite URLs that appear in the search results below.

""" + TRANSCRIPT_ERROR_NOTE + """

--- VIDEO NOTES ---
{notes}

--- WEB SEARCH RESULTS ---
{search_results}
"""

VERIFY_PROMPT = """You are a fact-checker. Below is a written section and \
the exact source material it's supposed to be based on. The video notes \
come from an auto-generated transcript that may itself contain \
speech-recognition errors in technical terms or brand names (e.g. "Jobex" \
instead of "Zabbix") - if the section uses a corrected, real term where the \
notes have an obviously garbled one, that's fine, leave it as-is.

Check EVERY factual claim in the section against the source material. For \
each claim:
- If it's directly supported by the video notes or the web search results \
(including a reasonable ASR correction), keep it as-is.
- If it is NOT supported by either source (invented, exaggerated, or from \
outside knowledge), remove that specific claim or reword the sentence to \
only state what the sources actually say. Do not add any new information \
that isn't in the sources.
- Only keep a source URL in the "Sources:" list if it appears in the web \
search results below.

Output ONLY the corrected section in the exact same format (### heading, \
explanation, **Sources:** list) - no commentary about what you changed, no \
preamble.

--- SECTION TO CHECK ---
{section}

--- SOURCE: VIDEO NOTES ---
{notes}

--- SOURCE: WEB SEARCH RESULTS ---
{search_results}
"""

VIDEO_ID_PATTERNS = [
    r"(?:v=|/)([0-9A-Za-z_-]{11})(?:[&?/]|$)",
    r"youtu\.be/([0-9A-Za-z_-]{11})",
]


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


def _transcript_api() -> YouTubeTranscriptApi:
    # Optional: route transcript fetches through a proxy (e.g. to work around
    # YouTube blocking cloud-host IPs). PROXY_URL should look like
    # http://user:pass@host:port. Note: free/datacenter proxies (tested with
    # Webshare's free tier) do NOT work here - YouTube blocks datacenter IP
    # ranges as a class. This only helps with a paid *residential* proxy.
    proxy_url = os.environ.get("PROXY_URL")
    if proxy_url:
        return YouTubeTranscriptApi(proxy_config=GenericProxyConfig(http_url=proxy_url, https_url=proxy_url))
    return YouTubeTranscriptApi()


def get_transcript(video_id: str) -> str:
    fetched = _transcript_api().fetch(video_id)
    return " ".join(snippet.text for snippet in fetched)


def build_notes(transcript: str, model: str) -> str:
    """Condense a transcript into notes usable for topic research. Chunks long transcripts (map-reduce)."""
    if len(transcript) <= CHUNK_CHARS:
        return transcript
    chunks = [transcript[i:i + CHUNK_CHARS] for i in range(0, len(transcript), CHUNK_CHARS)]
    print(f"  [condensing {len(chunks)} parts in parallel]", file=sys.stderr)
    with ThreadPoolExecutor(max_workers=CHUNK_WORKERS) as pool:
        parts = list(pool.map(lambda c: ask(model, CHUNK_SUMMARY_PROMPT.format(chunk=c)), chunks))
    return "\n\n".join(parts)[:COMBINED_NOTES_CHARS]


def extract_topics(notes: str, model: str) -> list:
    raw = ask(
        model,
        TOPIC_EXTRACTION_PROMPT.format(max_topics=MAX_TOPICS),
        context="VIDEO NOTES:\n\n" + notes[:CONTEXT_CHARS],
    )
    topics = []
    for line in raw.splitlines():
        line = line.strip().lstrip("-*").strip()
        line = line.lstrip("0123456789.)").strip()
        if line:
            topics.append(line)
    if not topics:
        topics = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    return topics[:MAX_TOPICS]


def research_topic(topic: str, notes: str, model: str) -> str:
    print(f"  [researching] {topic}", file=sys.stderr)
    image_pool = ThreadPoolExecutor(max_workers=1)
    image_future = image_pool.submit(image_search, topic)

    search_raw = web_search(topic, max_results=SEARCH_RESULTS_PER_TOPIC)
    print(f"    web_search -> {len(search_raw)} chars of results", file=sys.stderr)

    fetched_chunks = []
    urls_seen = 0
    for line in search_raw.splitlines():
        line = line.strip()
        if line.startswith("URL:") and urls_seen < PAGES_TO_FETCH_PER_TOPIC:
            url = line.split("URL:", 1)[1].strip()
            print(f"    fetch_page -> {url}", file=sys.stderr)
            page_text = fetch_page(url)
            fetched_chunks.append(f"[Fetched from {url}]\n{page_text}")
            urls_seen += 1

    search_results = search_raw
    if fetched_chunks:
        search_results += "\n\n--- FULL PAGE EXCERPTS ---\n\n" + "\n\n".join(fetched_chunks)

    notes_excerpt = notes[:CONTEXT_CHARS]
    search_excerpt = search_results[:CONTEXT_CHARS]

    prompt = TOPIC_WRITE_PROMPT.format(topic=topic, notes=notes_excerpt, search_results=search_excerpt)
    section = ask(model, prompt)

    print(f"    [verifying] {topic}", file=sys.stderr)
    verify_prompt = VERIFY_PROMPT.format(section=section, notes=notes_excerpt, search_results=search_excerpt)
    verified = ask(model, verify_prompt)

    image_url = image_future.result()
    image_pool.shutdown()
    if image_url:
        heading = f"### {topic}"
        verified = verified.replace(heading, f"{heading}\n\n![{topic}]({image_url})", 1)
    return verified


def summarize(transcript: str, model: str) -> str:
    notes = build_notes(transcript, model)

    print("  [identifying topics]", file=sys.stderr)
    topics = extract_topics(notes, model)
    if not topics:
        raise RuntimeError("Model didn't return any topics - try a different --model.")
    print(f"  Topics: {', '.join(topics)}", file=sys.stderr)

    with ThreadPoolExecutor(max_workers=TOPIC_RESEARCH_WORKERS) as pool:
        overview_future = pool.submit(ask, model, OVERVIEW_PROMPT, "VIDEO NOTES:\n\n" + notes[:CONTEXT_CHARS])
        sections = list(pool.map(lambda t: research_topic(t, notes, model), topics))
        overview = overview_future.result()

    return f"**About this video:** {overview}\n\n" + "\n\n".join(sections) + "\n"


def run(url: str, model: str = DEFAULT_MODEL, transcript_text: str = None) -> str:
    video_id = extract_video_id(url)
    print(f"Fetching video info for {video_id} ...", file=sys.stderr)
    meta = get_metadata(url)

    if transcript_text and transcript_text.strip():
        print("Using pasted transcript ...", file=sys.stderr)
        transcript = transcript_text.strip()
    else:
        print("Fetching transcript ...", file=sys.stderr)
        try:
            transcript = get_transcript(video_id)
        except Exception as e:
            raise RuntimeError(
                f"Couldn't get a transcript for this video ({e}). "
                "It may have captions disabled, or automatic fetching may be "
                "blocked here - try pasting the transcript text instead "
                "(copy it from YouTube's own \"Show transcript\" button)."
            )

    if not transcript.strip():
        raise RuntimeError("Transcript came back empty - this video may have no captions.")

    print(f"Summarizing + researching ({len(transcript):,} characters of transcript) ...", file=sys.stderr)
    summary = summarize(transcript, model)

    thumbnail_md = f"![{meta['title']}]({meta['thumbnail']})\n\n" if meta.get("thumbnail") else ""

    return (
        f"# {meta['title']}\n\n"
        f"{thumbnail_md}"
        f"**Channel:** {meta['author']}  \n"
        f"**URL:** {url}\n\n"
        f"{summary}\n"
    )


def main():
    parser = argparse.ArgumentParser(description="Summarize a YouTube video using Groq's free API + its transcript + free web research.")
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Groq model to use (default: {DEFAULT_MODEL})")
    parser.add_argument(
        "--transcript-file", type=str, default=None,
        help="Path to a text file with a manually-pasted transcript (skips auto-fetching)",
    )
    args = parser.parse_args()

    transcript_text = None
    if args.transcript_file:
        with open(args.transcript_file, encoding="utf-8") as f:
            transcript_text = f.read()

    try:
        report = run(args.url, args.model, transcript_text=transcript_text)
    except Exception as e:
        sys.exit(str(e))

    print(report)


if __name__ == "__main__":
    main()
