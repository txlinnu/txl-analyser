"""
PDF Research Agent - Free / Cloud Edition
------------------------------------------
$0 cost: uses Groq's free API (fast hosted inference, generous free tier,
no credit card) instead of a local model, so it doesn't touch your PC's
CPU/RAM - and free DuckDuckGo search instead of a paid web search tool.

Give it a PDF. It:
  1. Extracts the text from the PDF.
  2. Asks the model to identify the main topics.
  3. For EVERY topic, deterministically runs a real DuckDuckGo search and
     fetches the top result's page content - this step does not depend on
     the model "deciding" to call a tool, so it always actually happens
     (models often skip searching on topics they already "know"
     something about, which produces confident-sounding but unverified
     claims. Forcing the search step guarantees it's grounded in real,
     current web content instead).
  4. Writes a short, easy-to-understand summary of every topic - PDF
     content + real search findings, with real source links - saved as a
     Markdown file next to the PDF.

Note: unlike the fully-local version, your PDF text is sent to Groq's
servers to be processed (not 100% private) - that's the trade-off for
not loading down your own machine.

Setup (one-time):
    Get a free API key: https://console.groq.com/keys (no credit card)
    Put it in a .env file next to this script: GROQ_API_KEY=gsk_...
    pip install -r requirements-local.txt

Usage:
    python pdf_research_agent_local.py path/to/file.pdf
    python pdf_research_agent_local.py file.pdf --model openai/gpt-oss-20b
"""

import argparse
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from groq import Groq, RateLimitError
from pypdf import PdfReader

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DEFAULT_MODEL = "openai/gpt-oss-120b"
MAX_PDF_CHARS = 40_000
MAX_TOPICS = 6
SEARCH_RESULTS_PER_TOPIC = 4
PAGES_TO_FETCH_PER_TOPIC = 1
TOPIC_RESEARCH_WORKERS = 2  # topics researched in parallel (kept low - free tier has a low tokens/minute cap)
CONTEXT_CHARS = 3500  # PDF excerpt / search results chars per prompt (keeps calls under the free-tier TPM limit)
DOC_CONTEXT_CHARS = 6000  # PDF text used for the one-off topic-extraction / overview calls
MAX_RETRIES = 5

TOPIC_EXTRACTION_PROMPT = """You will be given the extracted text of a PDF \
document. List the main topics/subjects it covers, from most to least \
important - at most {max_topics}, but FEWER if the document doesn't \
support that many distinct topics. Do not pad the list or split one idea \
into multiple near-duplicate topics just to reach {max_topics}: a short or \
narrow document might genuinely only have 1-3 real topics, and that's \
fine. Each topic must be clearly distinct from the others.

Output ONLY a numbered list, one short topic name per line (3-6 words \
each), nothing else - no preamble, no explanation.

Example output:
1. Coral bleaching
2. Reef biodiversity
3. Climate change impact
"""

OVERVIEW_PROMPT = """Write a single short paragraph (2-4 sentences, plain \
language, no jargon) summarizing what this PDF document is about overall. \
Output only that paragraph, nothing else."""

TOPIC_WRITE_PROMPT = """You are writing one section of a research report \
about the topic: "{topic}"

Below is (1) the relevant excerpt from a PDF document, and (2) real, \
current web search results about this topic gathered just now.

Write a short section in this exact format:

### {topic}

<2-5 sentences in plain, jargon-free language that combines what the PDF \
says with what the web search found. If the web results add nothing new, \
just summarize the PDF. If they conflict with the PDF, briefly say so.>

**Sources:**
<a markdown bullet list of the URLs you actually used from the search \
results below - copy them exactly. If nothing from the web was useful, \
write "PDF only".>

Do not invent URLs. Only cite URLs that appear in the search results below.

--- PDF EXCERPT ---
{pdf_excerpt}

--- WEB SEARCH RESULTS ---
{search_results}
"""

VERIFY_PROMPT = """You are a fact-checker. Below is a written section and \
the exact source material it's supposed to be based on.

Check EVERY factual claim in the section against the source material. For \
each claim:
- If it's directly supported by the PDF excerpt or the web search results, \
keep it as-is.
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

--- SOURCE: PDF EXCERPT ---
{pdf_excerpt}

--- SOURCE: WEB SEARCH RESULTS ---
{search_results}
"""

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


def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web for a query using DuckDuckGo and return the top results.

    Args:
      query (str): The search query.
      max_results (int): How many results to return (default 5, max 10).

    Returns:
      str: A formatted list of results, each with a title, URL, and short snippet.
    """
    max_results = max(1, min(int(max_results), 10))
    last_error = None
    for attempt in range(2):  # one retry - DDGS backends sometimes fail transiently
        try:
            results = DDGS().text(query, max_results=max_results)
            break
        except Exception as e:
            last_error = e
            print(f"[web_search] attempt {attempt + 1} failed for {query!r}: {e}", file=sys.stderr)
            if attempt == 0:
                time.sleep(1.5)
    else:
        return f"Search failed: {last_error}"
    if not results:
        return "No results found."
    lines = []
    for r in results:
        lines.append(f"- {r.get('title', '')}\n  URL: {r.get('href', '')}\n  {r.get('body', '')}")
    return "\n".join(lines)


def image_search(query: str):
    """Find one relevant image for a topic. Returns a URL, or None on failure."""
    try:
        results = DDGS().images(query, max_results=1)
        return results[0]["image"] if results else None
    except Exception:
        return None


def fetch_page(url: str) -> str:
    """
    Fetch a web page and return its main readable text content.

    Args:
      url (str): The URL to fetch.

    Returns:
      str: Extracted, truncated text content of the page.
    """
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (research-agent)"},
            timeout=10,
        )
        resp.raise_for_status()
    except Exception as e:
        return f"Fetch failed: {e}"

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = " ".join(soup.get_text(separator=" ").split())
    return text[:3000]


def extract_pdf_text(pdf_path: Path):
    reader = PdfReader(str(pdf_path))
    pages = [page.extract_text() or "" for page in reader.pages]
    full_text = "\n\n".join(pages)
    if len(full_text.strip()) == 0:
        raise ValueError(
            "No extractable text found in this PDF. It may be a scanned "
            "image PDF that needs OCR first."
        )
    truncated = False
    if len(full_text) > MAX_PDF_CHARS:
        full_text = full_text[:MAX_PDF_CHARS]
        truncated = True
    return full_text, truncated


def _wait_seconds_from_error(e: Exception) -> float:
    m = re.search(r"try again in ([\d.]+)s", str(e))
    return float(m.group(1)) + 0.5 if m else 3.0


def ask(model: str, prompt: str, context: str = None) -> str:
    messages = []
    if context:
        messages.append({"role": "user", "content": context})
    messages.append({"role": "user", "content": prompt})

    for attempt in range(MAX_RETRIES):
        try:
            response = _get_client().chat.completions.create(
                model=model, messages=messages, temperature=0.3
            )
            return (response.choices[0].message.content or "").strip()
        except RateLimitError as e:
            if attempt == MAX_RETRIES - 1:
                raise
            wait = _wait_seconds_from_error(e)
            print(f"    [rate limited, waiting {wait:.1f}s]", file=sys.stderr)
            time.sleep(wait)


def extract_topics(pdf_text: str, model: str) -> list:
    raw = ask(
        model,
        TOPIC_EXTRACTION_PROMPT.format(max_topics=MAX_TOPICS),
        context="PDF TEXT:\n\n" + pdf_text[:DOC_CONTEXT_CHARS],
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


def research_topic(topic: str, pdf_text: str, model: str) -> str:
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

    pdf_excerpt = pdf_text[:CONTEXT_CHARS]
    search_excerpt = search_results[:CONTEXT_CHARS]

    prompt = TOPIC_WRITE_PROMPT.format(topic=topic, pdf_excerpt=pdf_excerpt, search_results=search_excerpt)
    section = ask(model, prompt)

    print(f"    [verifying] {topic}", file=sys.stderr)
    verify_prompt = VERIFY_PROMPT.format(section=section, pdf_excerpt=pdf_excerpt, search_results=search_excerpt)
    verified = ask(model, verify_prompt)

    image_url = image_future.result()
    image_pool.shutdown()
    if image_url:
        heading = f"### {topic}"
        verified = verified.replace(heading, f"{heading}\n\n![{topic}]({image_url})", 1)
    return verified


def run_agent(pdf_text: str, model: str) -> str:
    print("Identifying main topics ...", file=sys.stderr)
    topics = extract_topics(pdf_text, model)
    if not topics:
        raise RuntimeError("Model didn't return any topics - try a different --model.")
    print(f"  Topics: {', '.join(topics)}", file=sys.stderr)

    with ThreadPoolExecutor(max_workers=TOPIC_RESEARCH_WORKERS) as pool:
        overview_future = pool.submit(ask, model, OVERVIEW_PROMPT, "PDF TEXT:\n\n" + pdf_text[:DOC_CONTEXT_CHARS])
        sections = list(pool.map(lambda t: research_topic(t, pdf_text, model), topics))
        overview = overview_future.result()

    return overview + "\n\n" + "\n\n".join(sections) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="Summarize and research a PDF's topics using Groq's free API + free web search."
    )
    parser.add_argument("pdf", type=Path, help="Path to the PDF file")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Groq model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output markdown file path")
    args = parser.parse_args()

    if not args.pdf.exists():
        sys.exit(f"File not found: {args.pdf}")

    print(f"Reading {args.pdf} ...", file=sys.stderr)
    pdf_text, truncated = extract_pdf_text(args.pdf)
    if truncated:
        print(f"  Note: PDF text was truncated to {MAX_PDF_CHARS:,} characters for this run.", file=sys.stderr)

    print(f"Running agent (model={args.model}) ...", file=sys.stderr)
    try:
        report = run_agent(pdf_text, args.model)
    except RuntimeError as e:
        sys.exit(str(e))

    output_path = args.output or args.pdf.with_suffix(".summary.md")
    output_path.write_text(report, encoding="utf-8")

    print(f"\nDone. Report saved to {output_path}\n", file=sys.stderr)
    print(report)


if __name__ == "__main__":
    main()
