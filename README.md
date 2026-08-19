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

## TXL Cloud (chat assistant)

A separate, standalone chat app — [`chat_app.py`](chat_app.py) — Claude-style
multi-turn conversation, streamed replies, sidebar with chat history,
syntax-highlighted code with a copy button. It runs as its **own process
on its own port** so it doesn't clash with `app.py`, and has **two
interchangeable backends**:

| | Groq (default) | Ollama (local) |
|---|---|---|
| Agent | [`txl_cloud.py`](txl_cloud.py) | [`txl_cloud_local.py`](txl_cloud_local.py) |
| Cost | Free | Free |
| Speed | Fast (cloud) | As fast as your own hardware |
| Daily limit | Yes — Groq's free-tier token cap | **None** |
| Privacy | Messages sent to Groq's servers | **Nothing ever leaves your machine** |
| Setup | Free `GROQ_API_KEY` (already set up) | Install [Ollama](https://ollama.com/download) + pull a model |

**Heads up**: by default the Groq backend shares `GROQ_API_KEY` with `app.py`
(TXL Analyser) — both draw from the same daily free-tier quota, so heavy
use of one eats into the other's headroom. To give TXL Cloud its own
separate quota, create a second (free) Groq account and set
`CHAT_GROQ_API_KEY` in `.env` — see `.env.example`. Multiple keys on the
*same* Groq account still share one quota, so it has to be a different
account.

### Groq backend (fast, free, has a daily cap)

```bash
python chat_app.py
```
Open **http://127.0.0.1:5001**.

### Ollama backend (unlimited, fully private, needs decent hardware)

One-time setup, on whichever machine will run it:
1. Install Ollama: https://ollama.com/download
2. Pull a model:
   ```bash
   ollama pull qwen2.5:7b
   ollama pull qwen2.5:14b   # optional - bigger/more accurate, slower
   ```
3. `pip install -r requirements-local.txt` (includes the `ollama` package)

Then run:

**bash:**
```bash
CHAT_BACKEND=ollama CHAT_PORT=5002 python chat_app.py
```
**PowerShell:**
```powershell
$env:CHAT_BACKEND='ollama'; $env:CHAT_PORT='5002'; python chat_app.py
```

Open **http://127.0.0.1:5002**. Sized for ~32GB RAM + a small (~4GB) GPU —
Ollama automatically offloads what fits onto the GPU and runs the rest on
CPU. Expect noticeably slower replies than Groq (a few tokens/second on
that kind of hardware vs near-instant) — that's the trade for unlimited
and fully private. If it feels slow, use the "Balanced" (`qwen2.5:7b`)
model in the dropdown instead of "Accurate" (`qwen2.5:14b`).

### Notes for both

Conversation history is kept in memory only, tied to your browser
session — nothing written to disk. Click "New chat" to start a fresh
thread (past ones stay in the sidebar until the process restarts).

Sidebar extras:
- **Pin** a chat (📌 on hover) to keep it at the top, or **📁 move it into
  a Project** to group related chats — make a project with the **+** next
  to "Projects".
- **🗄 Artifacts** (in the sidebar) is a gallery of every code block 5+
  lines long TXL Cloud has written this session, across all chats and
  Code mode, with copy/download.
- **🎨 Customize** lets you set standing instructions (tone, preferred
  language, etc.) that get added to every message in Chat mode - saved
  in your browser only. Doesn't apply to Code mode, which already has its
  own specialized instructions.

You can run `app.py` (port 5000), `chat_app.py` (Groq, port 5001), and a
second `chat_app.py` (Ollama, port 5002) all at once in separate
terminals — the "← TXL Analyser" link in chat's sidebar points back to
`app.py`. If you change ports, set `ANALYSER_URL` (for chat_app.py) or
edit the link in `index.html` (for app.py) to match.

### Code mode - a small coding agent

Click the **"⌨ Code"** tab in TXL Cloud's sidebar (or go to `/code`) for
a real coding-agent mode, similar in spirit to Claude Code itself: the
model can list directories, read files, write files, and run shell
commands — using [`code_agent.py`](code_agent.py) and whichever chat
backend (Groq or Ollama) that instance is running.

**Safety model:**
- Everything is confined to a sandboxed workspace folder
  (`workspace/` next to this file by default — point it at a real project
  by setting `CODE_WORKSPACE` to that folder's path). Paths that try to
  escape it are rejected.
- `list_directory` / `read_file` run automatically (read-only).
- `write_file` / `run_command` **always** stop and show you exactly what's
  about to happen first — a diff for file writes, the exact command for
  shell commands — and wait for you to click **Approve** or **Deny**.
  Nothing touches disk or runs until you approve it.

One real limitation worth knowing: small/free models (especially local
ones like `qwen2.5:7b`) can occasionally *claim* they ran a command or
wrote a file in a compound request without actually calling the tool —
the system prompt tells it not to do this, but it isn't foolproof with
smaller models. Nothing unapproved ever actually executes either way
(the approval gate isn't something the model can talk its way around) —
just don't take a wordy "done!" at face value if no action card or
approval prompt showed up for it. Simple, one-thing-at-a-time requests
are noticeably more reliable than asking for several actions at once.

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
