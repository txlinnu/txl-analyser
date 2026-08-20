# Deploying for free (Render)

`render.yaml` defines **two** free web services - deploy either or both:
- `txl-analyser` → `app.py` (PDF/YouTube summarizer)
- `txl-cloud` → `chat_app.py` (the chat assistant, with real accounts)

Both get a free `https://<name>.onrender.com` subdomain automatically -
reachable from anywhere, no separate domain purchase needed.

## Deploying TXL Cloud specifically

It has real accounts and persisted chats/projects (`models.py`), backed
by a database - SQLite locally, but Render's free tier **wipes its own
disk on every restart/redeploy**, so production needs a real external
database, or every deploy would silently delete everyone's account:

1. Create a free Postgres database at https://neon.tech (no credit card).
2. In Neon's dashboard, copy the connection string (starts `postgres://...`).
3. When creating the `txl-cloud` service on Render (steps below), set
   these environment variables:
   - `DATABASE_URL` — the Neon connection string from step 2
   - `SECRET_KEY` — any long random string (e.g. generate one with
     `python -c "import secrets; print(secrets.token_hex(32))"`) -
     without this, sessions use an insecure default and everyone gets
     logged out on every redeploy
   - `GROQ_API_KEY` — your Groq key
   - `CHAT_GROQ_API_KEY` — optional, only if you set up the separate
     Groq account discussed earlier, to keep its quota independent of
     TXL Analyser's
   - `SITE_PASSWORD` — optional extra password gate on top of accounts
   - `SIGNUP_INVITE_CODE` — optional: requires this code to sign up, so
     having `SITE_PASSWORD` alone doesn't let anyone create their own
     account (and use Code mode's `run_command` on this server)
   - `SMTP_HOST` / `SMTP_USER` / `SMTP_PASSWORD` — optional, enables the
     "Forgot password" email. Any SMTP account works, e.g. a free Gmail
     account with an **app password** (not your normal login password) -
     generate one at https://myaccount.google.com/apppasswords. Without
     these set, "Forgot password" tells the user it isn't configured
     instead of pretending to send an email.
   - `GEMINI_API_KEY` — optional automatic fallback: if Groq's daily free
     quota runs out, chats seamlessly switch to Gemini instead of erroring
     out for the rest of the day. Free key (no credit card):
     https://aistudio.google.com/apikey. Leave unset to skip - Groq
     requests just fail normally once exhausted, same as before.
   - `ADMIN_EMAILS` — optional, comma-separated emails that can see
     `/admin` (account/usage counts). Leave unset and nobody can, including
     you - the route 404s for everyone and no "Admin" link shows.

**Only the Groq backend can be deployed this way** - Ollama needs real
GPU/CPU compute running locally, which Render's free tier doesn't provide.

Everything's ready to deploy — `render.yaml` and `.gitignore` are
already set up. These last steps need your own accounts
(I can't create accounts or log in on your behalf), so here's exactly
what to do:

Auto-deploy on push is active once Render's GitHub App is properly
installed (not just OAuth-authorized) at github.com/apps/render, with
access granted to this repo.

## 1. Push the code to GitHub

This repo is already connected to
https://github.com/txlinnu/txl-analyser - just push the latest changes:
```bash
git push
```
(If asked, `git add` and `git commit` the changed files first.)

## 2. Deploy on Render

**Easiest: Blueprint deploy (creates both services at once)**
1. Create a free account at https://render.com (free web services
   historically don't require a credit card — you'll see at signup).
2. In the Render dashboard: **New +** → **Blueprint**, connect the
   `txl-analyser` GitHub repo. Render reads `render.yaml` and shows both
   `txl-analyser` and `txl-cloud` ready to create together.
3. Before deploying, fill in each service's environment variables (Render
   prompts for the `sync: false` ones from `render.yaml`):
   - **txl-analyser**: `GROQ_API_KEY`, and `SITE_PASSWORD` (**strongly
     recommended** — without one, your site is public and unauthenticated,
     anyone with the URL could burn through your free Groq quota)
   - **txl-cloud**: `GROQ_API_KEY`, `DATABASE_URL` (your Neon connection
     string), `SECRET_KEY` (a long random string), and optionally
     `CHAT_GROQ_API_KEY` / `SITE_PASSWORD` — see the TXL Cloud section
     above for details on each
4. Click **Apply** / **Deploy**.

**Alternative: create each service manually**, if you'd rather deploy
just one, or Blueprint isn't available — **New +** → **Web Service** per
app, filling in Build Command `pip install -r requirements-local.txt`,
Start Command from `render.yaml` (`gunicorn app:app ...` or
`gunicorn chat_app:app ...`), Instance Type **Free**, and the same env
vars listed above for that service.

Render will build and deploy — takes a few minutes the first time.
You'll get free URLs like `https://txl-analyser.onrender.com` and
`https://txl-cloud.onrender.com`.

## 3. YouTube summaries on the live site: a known, unavoidable limit — and the free workaround

**Tested extensively**: YouTube blocks automated transcript fetching from
any cloud server as a matter of policy (their 2026 anti-bot system) - not
specific to Render. We confirmed this with: Render directly, two
different Webshare free proxy IPs, and even Netlify's edge network (a
completely different, non-datacenter-blocklisted network) using a
purpose-built library that tried three separate fallback strategies -
all blocked with "Sign in to confirm you're not a bot." This isn't
fixable by switching hosts or using free proxies; only a paid rotating
residential proxy reliably gets around it.

**The free fix - manual transcript paste:** the YouTube form on the live
site has a collapsible **"Auto-fetch not working? Paste the transcript
instead"** option. Copy the transcript from YouTube's own "Show
transcript" button (under "···" below the video) and paste it in - same
data YouTube would have given an automated fetch anyway, same output
quality, and it always works since nothing is being fetched from YouTube
at all.

**Practical result:**
- ✅ PDF summarization — works fine on the free public deployment
- ✅ YouTube summarization — works via auto-fetch locally, or via paste
  on the live site
- ❌ YouTube auto-fetch specifically on the live site — blocked by
  YouTube, not fixable for free (use paste instead)

If you ever want auto-fetch working on the public site anyway, the code
already supports it — set a `PROXY_URL` environment variable in Render
(format `http://user:pass@host:port`) pointing at a **paid residential**
proxy service, and it'll be used automatically. Nothing to change in the
code.

## What to expect

- **Free forever**, no credit card, no domain purchase.
- **Cold starts**: the free tier "spins down" after 15 minutes of no
  traffic. The next visit takes about a minute to wake back up — normal
  for free hosting, not a bug.
- **If you set `SITE_PASSWORD`**: your browser will prompt for a
  username (leave blank or anything) and the password you chose.
- **Updating later**: any time you `git push` new changes to the
  connected branch, Render redeploys automatically.

## Verifying it works

Once deployed, open the Render URL. You should see the same TXL
Analyser page you've been testing locally. Try a PDF or YouTube link —
if it errors, check the **Logs** tab in the Render dashboard first.
