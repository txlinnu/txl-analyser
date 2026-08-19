"""
PDF Research Agent
-------------------
Give it a PDF. It:
  1. Extracts the text from the PDF.
  2. Identifies the main topics covered.
  3. Uses Claude's built-in web search tool to research and verify/expand
     each topic with up-to-date information from the web.
  4. Produces a short, easy-to-understand summary of every topic, with
     sources cited, and saves it as a Markdown file next to the PDF.

Usage:
    python pdf_research_agent.py path/to/file.pdf
    python pdf_research_agent.py path/to/file.pdf --max-searches 15 --model claude-sonnet-5

Requires:
    pip install -r requirements.txt
    ANTHROPIC_API_KEY environment variable set (or a .env file with it).
"""

import argparse
import os
import sys
from pathlib import Path

from pypdf import PdfReader
import anthropic

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

MAX_PDF_CHARS = 180_000  # keep well within context; ~45k tokens of source text

SYSTEM_PROMPT = """You are a research assistant. The user will give you the extracted \
text of a PDF document.

Your job:
1. Identify the main topics/subjects covered in the document.
2. For each topic, use the web_search tool to look up current, reliable \
information that adds context, verifies facts, or fills in gaps the PDF \
doesn't cover. Search as many times as needed (within your budget) to \
gather a well-rounded picture of each topic.
3. Write a final report in Markdown with this structure:
   - A one-paragraph overview of what the document is about.
   - One section per topic, each with:
       - A short heading.
       - A plain-language explanation (2-5 sentences, no jargon, as if \
explaining to a smart person outside the field) combining what the PDF \
says with what you found on the web.
       - A "Sources" line listing the web sources you used for that topic \
(as markdown links). If a point comes only from the PDF, note that instead.
4. Keep the whole report skimmable: short paragraphs, no walls of text, \
bold the key terms.

Be accurate. If the web search results conflict with the PDF or with each \
other, say so briefly instead of picking one silently."""


def extract_pdf_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append(text)
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


def run_agent(pdf_text: str, model: str, max_searches: int) -> str:
    client = anthropic.Anthropic()

    tools = [
        {"type": "web_search_20250305", "name": "web_search", "max_uses": max_searches}
    ]

    messages = [
        {
            "role": "user",
            "content": (
                "Here is the extracted text of a PDF document. Research and "
                "summarize it per your instructions.\n\n---\n\n" + pdf_text
            ),
        }
    ]

    final_text_parts = []

    while True:
        response = client.messages.create(
            model=model,
            max_tokens=8000,
            system=SYSTEM_PROMPT,
            messages=messages,
            tools=tools,
        )

        for block in response.content:
            if block.type == "text":
                final_text_parts.append(block.text)
            elif block.type == "server_tool_use" and block.name == "web_search":
                print(f"  [searching] {block.input.get('query', '')}", file=sys.stderr)

        if response.stop_reason == "pause_turn":
            # Long-running search turn paused by the API; send it back unchanged to continue.
            messages.append({"role": "assistant", "content": response.content})
            final_text_parts = []  # response will be resent in full next round
            continue

        break

    return "".join(final_text_parts)


def main():
    parser = argparse.ArgumentParser(description="Summarize and research a PDF's topics using Claude + web search.")
    parser.add_argument("pdf", type=Path, help="Path to the PDF file")
    parser.add_argument("--model", default=os.environ.get("AGENT_MODEL", "claude-sonnet-5"), help="Claude model to use")
    parser.add_argument("--max-searches", type=int, default=15, help="Max web searches per run (cost control)")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output markdown file path")
    args = parser.parse_args()

    if not args.pdf.exists():
        sys.exit(f"File not found: {args.pdf}")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit(
            "ANTHROPIC_API_KEY is not set.\n"
            "Get a key at https://console.anthropic.com/ and set it, e.g.:\n"
            "  PowerShell:  $env:ANTHROPIC_API_KEY = 'sk-ant-...'\n"
            "  or put ANTHROPIC_API_KEY=sk-ant-... in a .env file next to this script."
        )

    print(f"Reading {args.pdf} ...", file=sys.stderr)
    pdf_text, truncated = extract_pdf_text(args.pdf)
    if truncated:
        print(
            f"  Note: PDF text was truncated to {MAX_PDF_CHARS:,} characters for this run.",
            file=sys.stderr,
        )

    print(f"Running agent (model={args.model}, max_searches={args.max_searches}) ...", file=sys.stderr)
    report = run_agent(pdf_text, args.model, args.max_searches)

    output_path = args.output or args.pdf.with_suffix(".summary.md")
    output_path.write_text(report, encoding="utf-8")

    print(f"\nDone. Report saved to {output_path}\n", file=sys.stderr)
    print(report)


if __name__ == "__main__":
    main()
