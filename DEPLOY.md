# Deploying TXL Analyser for free (Render)

Everything's ready to deploy — `render.yaml`, `.gitignore`, and a local
git commit are already set up. These last steps need your own accounts
(I can't create accounts or log in on your behalf), so here's exactly
what to do:

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
   - `WEBSHARE_PROXY_USERNAME` / `WEBSHARE_PROXY_PASSWORD` — needed for
     YouTube summaries to work on the public deployment (see step 3
     below). Skip these if you only care about PDF summaries on the
     live site.
5. Click **Deploy Web Service**.

Render will build and deploy — takes a few minutes the first time.
You'll get a free URL like `https://txl-analyser.onrender.com`.

## 3. Fix YouTube summaries on the live site (Webshare proxy)

YouTube blocks transcript requests from cloud/datacenter IPs (Render's
included) — this affects every hosting provider, not something specific
to this app. The fix: route those specific requests through a free proxy.

1. Sign up free at https://www.webshare.io/ (no credit card).
2. In the Webshare dashboard, go to **Proxy** → **List** (or
   **Settings**) and find your **Proxy Username** and **Proxy Password**
   (not your Webshare account login — a separate generated
   username/password for the proxies themselves).
3. In Render, go to your service → **Environment**, and add:
   - `WEBSHARE_PROXY_USERNAME`
   - `WEBSHARE_PROXY_PASSWORD`
4. Render redeploys automatically when you save environment variable
   changes.

Note: Webshare's free tier is datacenter proxies (10 of them, 1GB/month),
not residential ones. It's worth trying since it's free and often works,
but YouTube could still occasionally block a given proxy IP too - there's
no 100% guarantee without a paid residential proxy. PDF summaries don't
need this at all; they work on the free deployment either way.

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
