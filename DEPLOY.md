# Deploying TXL Analyser for free (Render)

Everything's ready to deploy — `render.yaml`, `.gitignore`, and a local
git commit are already set up. These last steps need your own accounts
(I can't create accounts or log in on your behalf), so here's exactly
what to do:

Auto-deploy on push is active once Render's GitHub App is properly
installed (not just OAuth-authorized) at github.com/apps/render, with
access granted to this repo.

## 1. Push the code to GitHub

If you don't have a GitHub account yet, create one free at
https://github.com/signup

Then, from `D:\AI Agents`:

1. Go to https://github.com/new and create a new repository (any name,
   e.g. `txl-analyser`). Leave it empty — don't add a README.
2. Copy the commands GitHub shows you under "…or push an existing
   repository from the command line", something like:
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/txl-analyser.git
   git branch -M main
   git push -u origin main
   ```
3. Run those in a terminal from `D:\AI Agents`.

## 2. Deploy on Render

1. Create a free account at https://render.com (free web services
   historically don't require a credit card — you'll see at signup).
2. In the Render dashboard: **New +** → **Web Service**, and connect the
   repo you just pushed. (If Render offers **Blueprint** instead and
   auto-detects `render.yaml`, use that — it's the same result, just
   pre-fills the fields below for you.)
3. If filling in fields manually, use:
   - **Build Command**: `pip install -r requirements-local.txt`
   - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT --timeout 300 --workers 1`
   - **Instance Type**: **Free**
4. Add these environment variables:
   - `GROQ_API_KEY` — your Groq key (the same one from `.env`)
   - `SITE_PASSWORD` — **strongly recommended**: pick a password. Without
     one, your site is public and unauthenticated — anyone with the URL
     could use it and burn through your free Groq quota, or upload
     arbitrary files.
5. Click **Deploy Web Service**.

Render will build and deploy — takes a few minutes the first time.
You'll get a free URL like `https://txl-analyser.onrender.com`.

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
