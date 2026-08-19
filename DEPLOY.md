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
2. In the Render dashboard: **New +** → **Blueprint**.
3. Connect your GitHub account and pick the repo you just pushed.
   Render will detect `render.yaml` automatically.
4. It'll ask you to fill in the environment variables marked
   `sync: false`:
   - `GROQ_API_KEY` — your Groq key (the same one from `.env`)
   - `SITE_PASSWORD` — **strongly recommended**: pick a password. Without
     one, your site is public and unauthenticated — anyone with the URL
     could use it and burn through your free Groq quota, or upload
     arbitrary files.
5. Click **Apply** / **Deploy**.

Render will build and deploy — takes a few minutes the first time.
You'll get a free URL like `https://txl-analyser.onrender.com`.

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
