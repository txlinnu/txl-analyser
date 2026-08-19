# TXL Analyser

Summarizes PDFs and YouTube videos: main topics explained in plain
language, grounded in real web sources / real transcripts.

Three ways to use it:

| | [`app.py`](app.py) (web UI) | [`pdf_research_agent_local.py`](pdf_research_agent_local.py) + [`youtube_agent.py`](youtube_agent.py) (CLI) | [`pdf_research_agent.py`](pdf_research_agent.py) (Claude CLI) |
|---|---|---|---|
| Cost | **Free**, forever | **Free**, forever | Paid (Claude API + web search fees) |
| Runs on | Groq's free cloud API | Groq's free cloud API | Anthropic's cloud (Claude) |
| Setup | Free Groq API key | Free Groq API key | Anthropic API key |
| Privacy | PDF/transcript text sent to Groq | PDF/transcript text sent to Groq | PDF text sent to Anthropic |

All three are "free" in the sense of no local compute cost either — the
model runs on Groq's/Anthropic's servers, not your PC, so your machine
stays responsive while it works.

## Web UI (recommended)

A local website with a form for PDFs and one for YouTube links.

### Setup (one-time)

1. Get a free Groq API key (no credit card): https://console.groq.com/keys
2. Put it in a `.env` file in this folder (copy `.env.example`):
   ```
   GROQ_API_KEY=gsk_your-key-here
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements-local.txt
   ```

### Run it

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in your browser. It only listens on
`127.0.0.1` (localhost) — nothing on your network or the internet can
reach it.

Each form has a "Speed / accuracy" dropdown:
- **Balanced** (`openai/gpt-oss-120b`) — better quality, a bit slower
- **Fast** (`openai/gpt-oss-20b`) — quicker, less detailed

(Groq's model lineup changes over time — if a model stops working, run
`python -c "from groq import Groq; [print(m.id) for m in Groq().models.list().data]"`
to see what's currently available and update `DEFAULT_MODEL` / `MODEL_CHOICES`.)

### Deploying it for free (a real public URL)

Want this reachable from anywhere, not just your PC? See
**[DEPLOY.md](DEPLOY.md)** — deploys to Render's free tier, no credit
card, no domain purchase. `render.yaml` is already set up for it.

One known limitation on the public deployment: YouTube blocks automated
transcript fetching from any cloud server (tested extensively - this is
YouTube's own 2026 anti-bot system, not fixable by switching hosts or
using free proxies). **PDF summaries work fine on the live site either
way.** For YouTube, you have two free options:
1. Run it locally instead (`python app.py` on your PC) - auto-fetch
   works fine there.
2. On the live site, use the **"paste the transcript instead"** option
   in the YouTube form - copy the transcript from YouTube's own "Show
   transcript" button and paste it in. Same data, same output quality,
   works every time, no fetching involved.

Details in DEPLOY.md.

## Command line

### PDF summarizer

```bash
python pdf_research_agent_local.py path/to/file.pdf
python pdf_research_agent_local.py file.pdf --model openai/gpt-oss-20b -o my_summary.md
```

How it works: extracts PDF text → asks the model for the main topics →
for **every** topic, deterministically runs a real DuckDuckGo search and
fetches the top result pages (not left up to the model to decide — models
often skip searching on topics they think they already know, which
produces confident-sounding but made-up sources) → writes a short,
sourced section per topic → saves as `<yourfile>.summary.md`.

### YouTube summarizer

```bash
python youtube_agent.py "https://www.youtube.com/watch?v=..."
python youtube_agent.py "https://youtu.be/..." --model openai/gpt-oss-20b
```

How it works: fetches the video's title/channel (YouTube's public oEmbed,
no key needed) and its transcript (`youtube-transcript-api`, no key, no
video download) → summarizes it topic-by-topic, grounded strictly in what
was actually said (long transcripts are chunked and combined). Requires
the video to have captions (auto-generated captions work fine).

### Notes

- Scanned/image-only PDFs won't extract text — you'd need OCR first.
- Videos with captions disabled can't be summarized (no transcript to
  read).
- Not 100% private: PDF text and video transcripts are sent to Groq's
  servers to be processed. If you need everything to stay fully on your
  own machine instead (trading speed/your PC's resources for that), ask
  to switch back to the local-Ollama version of these scripts.
- No AI summary is ever *perfectly* accurate — these are grounded as
  tightly as possible (search results, transcripts, PDF text only, with
  explicit "don't guess / don't embellish" instructions), but a model can
  still misread nuance. The "Balanced" model is more reliable than "Fast".

---

## Paid / Claude Edition

A separate, higher-quality option using Anthropic's Claude API directly
(real agentic tool-calling loop, Claude decides what to search).

### Setup

```bash
pip install -r requirements.txt
```

Set your Anthropic API key (get one at https://console.anthropic.com/):

**PowerShell:**
```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."
```

**Or** add it to the same `.env` file used above.

### Usage

```bash
python pdf_research_agent.py path/to/file.pdf
python pdf_research_agent.py file.pdf --max-searches 20 --model claude-sonnet-5 -o my_summary.md
```

- `--max-searches` — cap on web searches per run (cost control; default 15)
- `--model` — which Claude model to use (default `claude-sonnet-5`)
- `-o / --output` — where to save the Markdown report

### Notes

- Costs: token usage for the model + **$10 per 1,000 web searches** (see
  [Anthropic's pricing](https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool)).
  `--max-searches` bounds this per run.
- Very large PDFs are truncated to ~180k characters to stay within context.
