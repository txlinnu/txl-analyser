"""
TXL Cloud - Web UI
------------------------------------
A standalone local Flask website for the TXL Cloud chat agent. Runs as
its own process on its own port, separate from app.py (TXL Analyser) - so
both can run side by side without a port clash.

Two interchangeable backends, picked with CHAT_BACKEND:
  - groq (default)  txl_cloud.py       - Groq's free cloud API. Fast, but
                                          capped at a daily free-tier token
                                          quota, and your messages are
                                          sent to Groq's servers.
  - ollama           txl_cloud_local.py - runs entirely on this machine via
                                          Ollama. No cap, nothing ever
                                          leaves this PC - but needs Ollama
                                          installed and a model pulled
                                          first, and is only as fast as
                                          this machine's hardware.

Real accounts: sign up / log in with an email + password. Chats,
projects, and Code-mode history are stored per-account in a database
(models.py) - SQLite locally by default, or Postgres (e.g. Neon) in
production via DATABASE_URL - so it all survives logins and restarts.

Run:
    python chat_app.py                                    # Groq, port 5001
    CHAT_BACKEND=ollama CHAT_PORT=5002 python chat_app.py  # Ollama, port 5002
(PowerShell: $env:CHAT_BACKEND='ollama'; $env:CHAT_PORT='5002'; python chat_app.py)
"""

import ast
import html as html_escape
import json
import operator
import os
import re
import secrets
import shutil
import sys
import traceback
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import markdown as md
from flask import Flask, Response, g, jsonify, redirect, render_template, request, session, stream_with_context, url_for
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from pygments.formatters import HtmlFormatter

import code_agent
import mailer
import models
import project_rag
import txl_gemini
from pdf_research_agent_local import web_search

app = Flask(__name__)

# In-memory rate limiting (fine for this app's single-process deployment) -
# guards the auth endpoints against brute-force/spam. Everything else is
# unlimited by default.
limiter = Limiter(get_remote_address, app=app, default_limits=[], storage_uri="memory://")

BACKEND = os.environ.get("CHAT_BACKEND", "groq").strip().lower()

if BACKEND == "ollama":
    import txl_cloud_local as agent
    FAST_MODEL = "qwen2.5:7b"
    ACCURATE_MODEL = "qwen2.5:14b"
    MODEL_CHOICES = [
        ("auto", "Auto — picks the right model per message"),
        (FAST_MODEL, "Balanced — quicker, still solid"),
        (ACCURATE_MODEL, "Accurate — bigger, slower"),
    ]
    HEADER_BADGE = "🔒 Local · Unlimited"
    LANDING_SUBTITLE = "Runs 100% on this machine via Ollama — no daily limit, nothing ever sent anywhere."
    FOOTNOTE = "Runs entirely locally via Ollama — no data ever leaves this machine, no usage limit."
else:
    import txl_cloud as agent
    ACCURATE_MODEL = "openai/gpt-oss-120b"
    FAST_MODEL = "openai/gpt-oss-20b"
    MODEL_CHOICES = [
        ("auto", "Auto — picks the right model per message"),
        (ACCURATE_MODEL, "Balanced — more accurate, slower"),
        (FAST_MODEL, "Fast — quicker, less detailed"),
    ]
    HEADER_BADGE = "Free · Groq-powered"
    LANDING_SUBTITLE = "Free, powered by Groq's open models."
    FOOTNOTE = "Messages are sent to Groq's API for a reply. Chats are saved to your account."

VALID_MODELS = {m for m, _ in MODEL_CHOICES}
_CODE_HINT_RE = re.compile(
    r"```|\bdef \b|\bfunction\b|\bclass \b|\bimport \b|SELECT .* FROM|\berror\b|\btraceback\b|\bdebug\b|\bfix\b",
    re.IGNORECASE,
)


def route_model(message: str) -> str:
    """Heuristic router for the 'Auto' model choice: short, simple-looking
    questions go to the fast model; anything long, multi-part, or
    code/debugging-shaped goes to the bigger/more accurate one. Pure text
    heuristics - no extra API call, so it adds no latency of its own."""
    text = message or ""
    looks_complex = (
        len(text.split()) > 40
        or text.count("?") > 1
        or bool(_CODE_HINT_RE.search(text))
        or bool(_COMPOUND_JOINER_RE.search(text))  # "X, and also Y" - a compound ask even with one "?"
    )
    return ACCURATE_MODEL if looks_complex else FAST_MODEL


# --- "Deep check" mode: self-critique + a second model's opinion + ---------
# automated arithmetic verification, reconciled into one final answer.
# Opt-in (the "Deep check" composer toggle) since it costs 2-3x the normal
# latency/API calls - see chat.html's deep-check-toggle.

_SAFE_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_SAFE_UNARYOPS = {ast.USub: operator.neg, ast.UAdd: operator.pos}


def _safe_eval(node):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_BINOPS:
        return _SAFE_BINOPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_UNARYOPS:
        return _SAFE_UNARYOPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("unsupported expression")


def safe_calculate(expr: str) -> float:
    """Evaluates a plain arithmetic expression safely - only numbers, +-*/,
    //, %, **, and parentheses. No names, calls, or attribute access, so
    this can't be used to execute arbitrary code."""
    return _safe_eval(ast.parse(expr, mode="eval"))


_MATH_CLAIM_RE = re.compile(r"([0-9][0-9\s+\-*/().]*[0-9)])\s*=\s*(-?[0-9]+(?:\.[0-9]+)?)\b")


def verify_math_claims(text: str) -> list:
    """Scans a draft answer for 'expr = number' claims and flags any where
    the stated result doesn't match what the expression actually computes -
    catches arithmetic slips a model made confidently but wrong."""
    notes = []
    for expr_str, claimed_str in _MATH_CLAIM_RE.findall(text):
        if not re.search(r"[+\-*/]", expr_str):
            continue  # a bare number, not actually an expression
        try:
            actual = safe_calculate(expr_str)
            claimed = float(claimed_str)
        except Exception:
            continue
        if abs(actual - claimed) > max(1e-6, abs(actual) * 1e-9):
            notes.append(f"{expr_str.strip()} actually equals {actual:g}, not {claimed_str} as stated.")
    return notes


def _get_second_opinion(history, primary_model, custom_instructions) -> str:
    """An independent model's answer to the same question, for Deep check's
    consensus step. Prefers genuine model diversity (Gemini on the Groq
    backend) over just a different size of the same model family."""
    if BACKEND == "ollama":
        alt_model = ACCURATE_MODEL if primary_model == FAST_MODEL else FAST_MODEL
        return "".join(agent.stream_reply(history, alt_model, custom_instructions=custom_instructions))
    if txl_gemini.is_configured():
        system_prompt = agent.SYSTEM_PROMPT
        if custom_instructions:
            system_prompt += (
                "\n\nThe user has also given you these standing preferences for how "
                "they'd like you to behave - follow them unless they conflict with "
                f"being safe or honest:\n{custom_instructions}"
            )
        return "".join(txl_gemini.stream_reply(agent.trim_history(history), system_prompt))
    alt_model = FAST_MODEL if primary_model == ACCURATE_MODEL else ACCURATE_MODEL
    return "".join(agent.stream_reply(history, alt_model, custom_instructions=custom_instructions))


def deep_check_reply(history, model, custom_instructions=None, image_data_url=None):
    """
    Draft -> verify any arithmetic -> get a second, independent model's
    opinion -> have the primary model reconcile everything into one final
    answer. Same yield shape as agent.stream_reply (text chunks) so it's a
    drop-in replacement at the call site.
    """
    draft = "".join(agent.stream_reply(
        history, model, custom_instructions=custom_instructions, image_data_url=image_data_url,
    )).strip()

    math_notes = verify_math_claims(draft)

    second_opinion = ""
    if not image_data_url:  # no second vision-capable model on either backend
        try:
            second_opinion = _get_second_opinion(history, model, custom_instructions)
        except Exception as e:
            print(f"[deep_check] second opinion failed: {e}", file=sys.stderr)

    parts = []
    if second_opinion:
        parts.append(f"An independent second model's answer to the same question:\n{second_opinion}")
    if math_notes:
        parts.append("Automated arithmetic check found possible issues:\n" + "\n".join(math_notes))
    review_note = (
        "\n\n".join(parts) + "\n\nReview your draft against the above and correct anything wrong."
        if parts else
        "Double-check your draft above for accuracy and completeness before finalizing."
    )

    last = history[-1]
    review_history = history[:-1] + [{
        "role": "user",
        "content": (
            last["content"]
            + f"\n\n[Internal review - your own draft answer was:]\n{draft}\n\n{review_note}\n\n"
            "(Give ONE final, clean answer. Don't mention this review process, the draft, "
            "or any other model to the user.)"
        ),
    }]
    yield from agent.stream_reply(
        review_history, model, custom_instructions=custom_instructions, image_data_url=image_data_url,
    )


MAX_SEARCH_SUBQUERIES = 3

# Catches "..., and what/when/where/who/why/how/is/does/can ..." - the
# common way people join two distinct questions under one final "?"
# without a second "?" to split on (e.g. "what's today's date, and what's
# a recent AI headline?" has exactly one "?", so naive splitting on "?"
# treats it as a single query and only ever searches the first half).
_COMPOUND_JOINER_RE = re.compile(
    r",?\s+and\s+(what|when|where|who|why|how|which|is|are|does|do|can|will)\b", re.IGNORECASE
)


def _split_into_queries(message: str) -> list:
    parts = [p.strip() for p in message.split("?") if p.strip()]
    if len(parts) > 1:
        return [p + "?" for p in parts]

    m = _COMPOUND_JOINER_RE.search(message)
    if m:
        first = message[:m.start()].strip()
        second = (m.group(1) + message[m.end():]).strip()
        candidates = [q for q in (first, second) if len(q) > 3]
        if len(candidates) > 1:
            return candidates

    return [message]


def web_search_for_message(message: str) -> str:
    """
    Deterministic web search for chat grounding. A single search call
    often misses part of a compound question - e.g. "what's today's date,
    and what's a recent AI headline?" as one DuckDuckGo query tends to
    return only date-related results, so the second half goes ungrounded.
    Split into distinct questions and search each separately when there's
    more than one; a plain single-topic message still gets a single
    search, unchanged from before.
    """
    queries = _split_into_queries(message)[:MAX_SEARCH_SUBQUERIES]

    if len(queries) == 1:
        try:
            return web_search(queries[0], max_results=5)
        except Exception as e:
            return f"Search failed: {e}"

    blocks = []
    for q in queries:
        try:
            result = web_search(q, max_results=4)
        except Exception as e:
            result = f"Search failed: {e}"
        blocks.append(f'Results for "{q}":\n{result}')
    return "\n\n".join(blocks)


MD_EXTENSIONS = ["extra", "sane_lists", "codehilite"]
MD_EXTENSION_CONFIGS = {"codehilite": {"guess_lang": False}}


def render_markdown(text: str) -> str:
    return md.markdown(text, extensions=MD_EXTENSIONS, extension_configs=MD_EXTENSION_CONFIGS)


# --- Artifacts: Claude-style side panel for big code blocks -----------------
# Any fenced code block of ARTIFACT_MIN_LINES+ lines gets pulled out of the
# chat bubble into a clickable card that opens in the side panel instead -
# short snippets stay inline as normal syntax-highlighted code.
ARTIFACT_MIN_LINES = 5
_FENCE_RE = re.compile(r"```([a-zA-Z0-9_+-]*)\n(.*?)```", re.DOTALL)
_LANG_EXT = {
    "python": "py", "py": "py", "javascript": "js", "js": "js",
    "typescript": "ts", "ts": "ts", "jsx": "jsx", "tsx": "tsx",
    "bash": "sh", "shell": "sh", "sh": "sh", "powershell": "ps1",
    "html": "html", "css": "css", "json": "json", "sql": "sql",
    "java": "java", "c": "c", "cpp": "cpp", "c++": "cpp", "go": "go",
    "rust": "rs", "ruby": "rb", "php": "php", "yaml": "yaml", "yml": "yaml",
    "markdown": "md", "md": "md", "csharp": "cs", "c#": "cs",
}


def _artifact_filename(language: str) -> str:
    ext = _LANG_EXT.get((language or "").lower(), "txt")
    return f"snippet.{ext}"


def _extract_artifacts(markdown_text: str):
    """Pull big fenced code blocks out into artifact dicts, leaving a marker behind."""
    artifacts = []

    def _replace(match):
        language = match.group(1).strip()
        code = match.group(2)
        code = code[:-1] if code.endswith("\n") else code
        line_count = code.count("\n") + 1
        if line_count < ARTIFACT_MIN_LINES:
            return match.group(0)  # short block - leave it inline, unchanged
        art_id = uuid.uuid4().hex[:10]
        artifacts.append({
            "id": art_id,
            "language": language or "text",
            "title": _artifact_filename(language),
            "code": code,
            "lines": line_count,
        })
        return f"\n\n[[ARTIFACT:{art_id}]]\n\n"

    processed = _FENCE_RE.sub(_replace, markdown_text)
    return processed, artifacts


# Artifacts in these languages get a live "Preview" tab (sandboxed iframe)
# alongside the code - anything else only shows the highlighted source.
RENDERABLE_ARTIFACT_LANGS = {"html", "svg"}


def _artifact_card_html(art: dict) -> str:
    code_attr = html_escape.escape(art["code"], quote=True)
    title_attr = html_escape.escape(art["title"], quote=True)
    lang_attr = html_escape.escape(art["language"], quote=True)
    title_txt = html_escape.escape(art["title"])
    lang_txt = html_escape.escape(art["language"])
    renderable = "true" if art["language"].lower() in RENDERABLE_ARTIFACT_LANGS else "false"
    highlighted = render_markdown(f"```{art['language']}\n{art['code']}\n```")
    return (
        f'<div class="artifact-card" data-id="{art["id"]}" data-title="{title_attr}" '
        f'data-lang="{lang_attr}" data-code="{code_attr}" data-renderable="{renderable}">'
        f'<div class="artifact-icon">{"▶" if renderable == "true" else "&lt;/&gt;"}</div>'
        f'<div class="artifact-meta">'
        f'<div class="artifact-title">{title_txt}</div>'
        f'<div class="artifact-sub">{lang_txt} · {art["lines"]} lines</div>'
        f'</div><div class="artifact-open">Open ›</div></div>'
        f'<template class="artifact-src" data-id="{art["id"]}">{highlighted}</template>'
    )


def render_message(content: str) -> str:
    """Render an assistant message to HTML, promoting big code blocks to artifact cards."""
    processed, artifacts = _extract_artifacts(content)
    html_out = render_markdown(processed)
    for art in artifacts:
        card_html = _artifact_card_html(art)
        marker_p = f"<p>[[ARTIFACT:{art['id']}]]</p>"
        if marker_p in html_out:
            html_out = html_out.replace(marker_p, card_html, 1)
        else:
            html_out = html_out.replace(f"[[ARTIFACT:{art['id']}]]", card_html, 1)
    return html_out


# Server-side code syntax highlighting (Pygments) - one light style, one dark,
# the dark one scoped behind prefers-color-scheme so it matches the page's
# own light/dark theme. nobackground=True: the page's own --code-bg wins.
PYGMENTS_CSS = (
    HtmlFormatter(style="xcode", nobackground=True).get_style_defs(".bubble .codehilite")
    + "\n@media (prefers-color-scheme: dark) {\n"
    + HtmlFormatter(style="monokai", nobackground=True).get_style_defs(".bubble .codehilite")
    + "\n}"
)

app.secret_key = os.environ.get("SECRET_KEY") or "dev-only-insecure-key-set-SECRET_KEY-in-production"
models.init_db(app)

# Optional extra password gate on TOP of accounts - set SITE_PASSWORD in the
# environment (e.g. on Render) to require a shared password before anyone
# can even reach the login page on a public deployment. Unset = no gate.
SITE_PASSWORD = os.environ.get("SITE_PASSWORD")

# Optional: require an invite code to sign up (on top of SITE_PASSWORD, if
# also set) - stops anyone who has SITE_PASSWORD from creating their own
# account and using Code mode's run_command on this server. Unset = signup
# stays open to anyone who reaches it, same as before.
SIGNUP_INVITE_CODE = os.environ.get("SIGNUP_INVITE_CODE")

# Where the "Analyser" link in the sidebar points - the other app (app.py),
# which normally runs on port 5000. Override with ANALYSER_URL if you run
# it elsewhere.
ANALYSER_URL = os.environ.get("ANALYSER_URL", "http://127.0.0.1:5000/")

# Optional: comma-separated emails allowed to see /admin (usage counts,
# recent signups). Unset = nobody can - the route 404s for everyone,
# admin link never shows, rather than defaulting to "whoever signed up
# first" (which could easily be wrong on an already-populated database).
ADMIN_EMAILS = {e.strip().lower() for e in os.environ.get("ADMIN_EMAILS", "").split(",") if e.strip()}

PUBLIC_ENDPOINTS = {"login", "signup", "static", "forgot_password", "reset_password", "service_worker"}


@app.route("/sw.js")
def service_worker():
    # Served from the root (not /static/sw.js) so its default scope covers
    # the whole app, not just /static/ - required for it to control /, /code, etc.
    # A plain Response (not send_from_directory's conditional/caching path)
    # avoids duplicate response headers that make browsers reject the SW fetch.
    sw_path = Path(app.static_folder) / "sw.js"
    return Response(sw_path.read_text(encoding="utf-8"), mimetype="application/javascript")


@app.before_request
def require_password():
    if not SITE_PASSWORD:
        return None
    auth = request.authorization
    if not auth or auth.password != SITE_PASSWORD:
        return Response(
            "Login required", 401, {"WWW-Authenticate": 'Basic realm="TXL Cloud"'}
        )
    return None


@app.before_request
def require_login():
    if request.endpoint in PUBLIC_ENDPOINTS or request.endpoint is None:
        return None
    user = models.User.query.get(session.get("user_id")) if session.get("user_id") else None
    if not user:
        session.pop("user_id", None)
        if request.method == "GET":
            return redirect(url_for("login", next=request.path))
        return jsonify({"error": "Please log in."}), 401
    g.user = user
    return None


@app.errorhandler(500)
def handle_unexpected_error(e):
    """
    Safety net for any route-level exception that happens outside a
    streaming generator's own try/except (see chat_send/chat_edit for the
    class of bug this guards against) - without this, Flask's default 500
    page is HTML, which the frontend's fetch().json() calls can't parse,
    surfacing only a generic "Something went wrong." with zero information
    about what actually failed. Every fetch() call this app makes sends
    Content-Type: application/json, so that's the reliable signal for
    "this needs a JSON error back" - anything else (a real page load) gets
    a minimal plain error page instead of Flask's default (which can leak
    internals in some configurations).
    """
    traceback.print_exc(file=sys.stderr)
    if request.is_json:
        return jsonify({"error": "Something went wrong on our end. Please try again."}), 500
    return Response("<h1>Something went wrong</h1><p>Please try again.</p>", status=500, mimetype="text/html")


def pick_model(data, message: str = "") -> str:
    model = data.get("model", "")
    if model not in VALID_MODELS:
        return agent.DEFAULT_MODEL
    if model == "auto":
        return route_model(message)
    return model


_IMAGE_DATA_URL_RE = re.compile(r"^data:image/(png|jpeg|jpg|webp|gif);base64,[A-Za-z0-9+/]+=*$")
MAX_IMAGE_B64_CHARS = 8_000_000  # ~6 MB raw image, base64-inflated


class InvalidImage(ValueError):
    pass


def _validate_image(image_data_url):
    """Returns the data URL unchanged if it's a well-formed, reasonably-sized
    pasted image - None if nothing was sent. Raises InvalidImage otherwise,
    so the route can fail fast with a 400 instead of forwarding junk to a
    model API."""
    if not image_data_url:
        return None
    if len(image_data_url) > MAX_IMAGE_B64_CHARS:
        raise InvalidImage("That image is too large (max ~6 MB).")
    if not _IMAGE_DATA_URL_RE.match(image_data_url):
        raise InvalidImage("Unsupported image data.")
    return image_data_url


def _make_title(message: str) -> str:
    title = " ".join(message.split())  # collapse whitespace/newlines
    return title if len(title) <= 48 else title[:47].rstrip() + "…"


def _delete_conversations(conv_ids):
    """
    Deletes Conversations by id, WITH their Messages. Bulk Query.delete()
    doesn't trigger the ORM's cascade="all, delete-orphan" (that only
    fires on individual db.session.delete(obj) calls) - skipping this and
    bulk-deleting Conversation alone would silently orphan its Messages.
    Since primary keys can be reused after a row is deleted (SQLite ROWID
    reuse), an orphaned Message can then resurface attached to a brand
    new, unrelated conversation that happens to get the same id.
    """
    conv_ids = [cid for cid in conv_ids if cid is not None]
    if not conv_ids:
        return
    models.Message.query.filter(models.Message.conversation_id.in_(conv_ids)).delete(synchronize_session=False)
    models.Conversation.query.filter(models.Conversation.id.in_(conv_ids)).delete(synchronize_session=False)


# --- Auth --------------------------------------------------------------

def _valid_email(email: str) -> bool:
    return "@" in email and "." in email.split("@")[-1] and len(email) <= 255


@app.route("/signup", methods=["GET", "POST"])
@limiter.limit("8 per hour", methods=["POST"])
def signup():
    if request.method == "GET":
        return render_template("signup.html", error=None, invite_required=bool(SIGNUP_INVITE_CODE))
    email = (request.form.get("email") or "").strip().lower()
    password = request.form.get("password") or ""
    confirm = request.form.get("confirm") or ""
    invite_code = request.form.get("invite_code") or ""
    invite_required = bool(SIGNUP_INVITE_CODE)
    if invite_required and invite_code != SIGNUP_INVITE_CODE:
        return render_template("signup.html", error="Incorrect invite code.", email=email, invite_required=True)
    if not _valid_email(email):
        return render_template("signup.html", error="Enter a valid email address.", email=email, invite_required=invite_required)
    if len(password) < 8:
        return render_template("signup.html", error="Password must be at least 8 characters.", email=email, invite_required=invite_required)
    if password != confirm:
        return render_template("signup.html", error="Passwords don't match.", email=email, invite_required=invite_required)
    if models.User.query.filter_by(email=email).first():
        return render_template("signup.html", error="An account with that email already exists.", email=email, invite_required=invite_required)

    user = models.User(email=email)
    user.set_password(password)
    models.db.session.add(user)
    models.db.session.commit()
    session["user_id"] = user.id
    return redirect(url_for("chat_home"))


@app.route("/login", methods=["GET", "POST"])
@limiter.limit("10 per minute; 30 per hour", methods=["POST"])
def login():
    if request.method == "GET":
        return render_template("login.html", error=None)
    email = (request.form.get("email") or "").strip().lower()
    password = request.form.get("password") or ""
    user = models.User.query.filter_by(email=email).first()
    if not user or not user.check_password(password):
        return render_template("login.html", error="Incorrect email or password.", email=email)
    session["user_id"] = user.id
    next_url = request.args.get("next") or url_for("chat_home")
    return redirect(next_url)


@app.route("/logout", methods=["POST"])
def logout():
    session.pop("user_id", None)
    return redirect(url_for("login"))


RESET_TOKEN_LIFETIME = timedelta(hours=1)


@app.route("/forgot-password", methods=["GET", "POST"])
@limiter.limit("5 per hour", methods=["POST"])
def forgot_password():
    if request.method == "GET":
        return render_template("forgot_password.html", error=None, sent=False, mail_configured=mailer.is_configured())

    if not mailer.is_configured():
        return render_template(
            "forgot_password.html", sent=False, mail_configured=False,
            error="Password reset email isn't set up on this deployment yet - ask whoever runs it to configure SMTP_HOST/SMTP_USER/SMTP_PASSWORD.",
        )

    email = (request.form.get("email") or "").strip().lower()
    user = models.User.query.filter_by(email=email).first()
    # Always show the same "sent" response whether or not the account exists,
    # so this can't be used to find out which emails have accounts here.
    if user:
        user.reset_token = secrets.token_urlsafe(32)
        user.reset_token_expires = datetime.now(timezone.utc) + RESET_TOKEN_LIFETIME
        models.db.session.commit()
        reset_url = url_for("reset_password", token=user.reset_token, _external=True)
        try:
            mailer.send_password_reset_email(user.email, reset_url)
        except Exception:
            traceback.print_exc(file=sys.stderr)
            # Don't leak send failures to the client - same generic message either way.
    return render_template("forgot_password.html", error=None, sent=True, mail_configured=True)


@app.route("/reset-password", methods=["GET", "POST"])
@limiter.limit("20 per hour", methods=["POST"])
def reset_password():
    token = request.values.get("token", "")
    user = models.User.query.filter_by(reset_token=token).first() if token else None
    valid = bool(
        user and user.reset_token_expires
        and user.reset_token_expires.replace(tzinfo=timezone.utc) > datetime.now(timezone.utc)
    )

    if request.method == "GET":
        return render_template("reset_password.html", token=token, valid=valid, error=None)

    if not valid:
        return render_template("reset_password.html", token=token, valid=False, error=None)

    password = request.form.get("password") or ""
    confirm = request.form.get("confirm") or ""
    if len(password) < 8:
        return render_template("reset_password.html", token=token, valid=True, error="Password must be at least 8 characters.")
    if password != confirm:
        return render_template("reset_password.html", token=token, valid=True, error="Passwords don't match.")

    user.set_password(password)
    user.reset_token = None
    user.reset_token_expires = None
    models.db.session.commit()
    return redirect(url_for("login"))


# --- Chat mode -----------------------------------------------------------

def _sidebar_data(user, mode: str, active_id=None) -> dict:
    """
    Groups this user's conversations OF ONE MODE ("chat" or "code") for the
    sidebar: pinned chats (regardless of project) first, then each
    project's own (non-pinned, same-mode) chats, then everything else
    ungrouped. Most-recently-created first. Projects themselves are shared
    across both modes - only the chat lists shown under them are filtered.
    """
    convs = models.Conversation.query.filter_by(user_id=user.id, mode=mode).order_by(models.Conversation.id.desc()).all()
    projects = models.Project.query.filter_by(user_id=user.id).order_by(models.Project.id.desc()).all()

    def _item(c):
        return {"id": c.id, "title": c.title, "active": c.id == active_id, "pinned": c.pinned}

    pinned = [_item(c) for c in convs if c.pinned]

    proj_list = []
    for p in projects:
        chats = [_item(c) for c in convs if c.project_id == p.id and not c.pinned]
        proj_list.append({"id": p.id, "name": p.name, "chats": chats})

    ungrouped = [_item(c) for c in convs if not c.pinned and not c.project_id]

    return {"pinned": pinned, "projects": proj_list, "ungrouped": ungrouped}


@app.route("/", methods=["GET"])
def chat_home():
    user = g.user
    conv_id = request.args.get("c", type=int)
    conv = models.Conversation.query.filter_by(id=conv_id, user_id=user.id, mode="chat").first() if conv_id else None

    if conv:
        msgs = _ordered_messages(conv.id)
        history = [
            {
                "id": m.id,
                "role": m.role,
                "html": render_message(m.content) if m.role == "assistant" else None,
                "text": m.content,
                "image_data": m.image_data,
                "is_last": i == len(msgs) - 1,
            }
            for i, m in enumerate(msgs)
        ]
    else:
        conv_id = None
        history = []

    sidebar = _sidebar_data(user, "chat", conv_id)
    return render_template(
        "chat.html",
        model_choices=MODEL_CHOICES,
        history=history,
        active_id=conv_id,
        pinned=sidebar["pinned"],
        projects=sidebar["projects"],
        ungrouped=sidebar["ungrouped"],
        analyser_url=ANALYSER_URL,
        pygments_css=PYGMENTS_CSS,
        header_badge=HEADER_BADGE,
        landing_subtitle=LANDING_SUBTITLE,
        footnote=FOOTNOTE,
        user_email=user.email,
        show_admin_link=is_admin(user),
    )


@app.route("/send", methods=["POST"])
def chat_send():
    user = g.user
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"error": "Please type a message."}), 400
    if len(message) > 8000:
        return jsonify({"error": "That message is too long (max 8000 characters)."}), 400
    try:
        image_data_url = _validate_image(data.get("image"))
    except InvalidImage as e:
        return jsonify({"error": str(e)}), 400

    model = pick_model(data, message)
    custom_instructions = (data.get("custom_instructions") or "").strip()[:4000] or None

    conv_id = data.get("conv_id")
    conv = models.Conversation.query.filter_by(id=conv_id, user_id=user.id, mode="chat").first() if conv_id else None
    is_new = conv is None
    if is_new:
        conv = models.Conversation(user_id=user.id, mode="chat", title=_make_title(message))
        models.db.session.add(conv)
        models.db.session.commit()

    user_msg = models.Message(conversation_id=conv.id, role="user", content=message, image_data=image_data_url)
    models.db.session.add(user_msg)
    models.db.session.commit()
    user_message_id = user_msg.id
    history = [{"role": m.role, "content": m.content} for m in conv.messages]
    conv_id, conv_title = conv.id, conv.title

    # Extra context folded into the API call only - the clean original
    # message is what's actually stored in the DB above.
    extra_blocks = []

    # Web search grounding, if the user toggled it on: deterministically search
    # (not left up to the model to decide - see pdf_research_agent_local.py for why).
    # This whole enrichment block runs before the SSE stream starts, so any
    # failure here has to be caught explicitly - it can't rely on generate()'s
    # try/except, and an uncaught exception here would surface as a raw Flask
    # 500 page (not JSON), which the frontend can only show as a generic
    # "Something went wrong." A failed enrichment should degrade gracefully
    # (skip that one piece), never take down the whole message.
    if data.get("web_search"):
        try:
            results = web_search_for_message(message)
            extra_blocks.append((
                "Web search results for grounding - answer EVERY part of the question "
                "above using the relevant results below; if a part genuinely isn't "
                "covered by any result, say so explicitly rather than skipping it. "
                "Cite the URLs you actually use, don't invent any",
                results,
            ))
        except Exception as e:
            traceback.print_exc(file=sys.stderr)

    # This chat's project (if any) may have reference files attached - fold
    # them in on every turn so they're not lost to history trimming.
    try:
        knowledge = _project_knowledge_block(conv.project_id, message)
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        knowledge = ""
    if knowledge:
        extra_blocks.append((
            "Reference material attached to this project - use it where relevant",
            knowledge,
        ))

    if extra_blocks:
        content = message
        for label, block in extra_blocks:
            content += f"\n\n[{label}]:\n{block}"
        history[-1] = {"role": "user", "content": content}

    reply_fn = deep_check_reply if data.get("deep_check") else agent.stream_reply

    def generate():
        if is_new:
            yield f"data: {json.dumps({'conv_id': conv_id, 'title': conv_title})}\n\n"
        yield f"data: {json.dumps({'user_message_id': user_message_id})}\n\n"
        chunks = []
        try:
            for delta in reply_fn(history, model, custom_instructions=custom_instructions, image_data_url=image_data_url):
                chunks.append(delta)
                yield f"data: {json.dumps({'delta': delta})}\n\n"
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            last = models.Message.query.filter_by(conversation_id=conv_id).order_by(models.Message.id.desc()).first()
            if last and last.role == "user":
                models.db.session.delete(last)
            if is_new:
                _delete_conversations([conv_id])
            models.db.session.commit()
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            return
        full_reply = "".join(chunks).strip()
        models.db.session.add(models.Message(conversation_id=conv_id, role="assistant", content=full_reply))
        models.db.session.commit()
        html_out = render_message(full_reply)
        yield f"data: {json.dumps({'done': True, 'html': html_out})}\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


def _ordered_messages(conv_id):
    return models.Message.query.filter_by(conversation_id=conv_id).order_by(models.Message.id).all()


@app.route("/conv/<int:conv_id>/regenerate", methods=["POST"])
def chat_regenerate(conv_id):
    conv = models.Conversation.query.filter_by(id=conv_id, user_id=g.user.id, mode="chat").first()
    if not conv:
        return jsonify({"error": "Not found"}), 404
    msgs = _ordered_messages(conv_id)
    if not msgs or msgs[-1].role != "assistant":
        return jsonify({"error": "Nothing to regenerate."}), 400

    data = request.get_json(silent=True) or {}
    last_user_text = msgs[-2].content if len(msgs) >= 2 and msgs[-2].role == "user" else ""
    model = pick_model(data, last_user_text)
    custom_instructions = (data.get("custom_instructions") or "").strip()[:4000] or None
    old_assistant_id = msgs[-1].id
    history = [{"role": m.role, "content": m.content} for m in msgs[:-1]]
    image_data_url = msgs[-2].image_data if len(msgs) >= 2 and msgs[-2].role == "user" else None

    def generate():
        chunks = []
        try:
            for delta in agent.stream_reply(history, model, custom_instructions=custom_instructions, image_data_url=image_data_url):
                chunks.append(delta)
                yield f"data: {json.dumps({'delta': delta})}\n\n"
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            return
        # Only replace the old reply once the new one actually succeeded -
        # a failed regenerate should never leave the user with nothing.
        full_reply = "".join(chunks).strip()
        old = models.Message.query.get(old_assistant_id)
        if old:
            models.db.session.delete(old)
        models.db.session.add(models.Message(conversation_id=conv_id, role="assistant", content=full_reply))
        models.db.session.commit()
        html_out = render_message(full_reply)
        yield f"data: {json.dumps({'done': True, 'html': html_out})}\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


@app.route("/conv/<int:conv_id>/edit", methods=["POST"])
def chat_edit(conv_id):
    conv = models.Conversation.query.filter_by(id=conv_id, user_id=g.user.id, mode="chat").first()
    if not conv:
        return jsonify({"error": "Not found"}), 404
    data = request.get_json(silent=True) or {}
    new_text = (data.get("message") or "").strip()
    if not new_text:
        return jsonify({"error": "Please type a message."}), 400
    if len(new_text) > 8000:
        return jsonify({"error": "That message is too long (max 8000 characters)."}), 400

    target = models.Message.query.filter_by(id=data.get("message_id"), conversation_id=conv_id, role="user").first()
    if not target:
        return jsonify({"error": "Message not found."}), 404

    model = pick_model(data, new_text)
    custom_instructions = (data.get("custom_instructions") or "").strip()[:4000] or None

    # Editing a past message discards it and everything after it (this
    # branch of the conversation), then continues from the edited version -
    # same as real Claude/ChatGPT's "edit and resend" behavior.
    for m in models.Message.query.filter(
        models.Message.conversation_id == conv_id, models.Message.id >= target.id
    ).all():
        models.db.session.delete(m)
    models.db.session.add(models.Message(conversation_id=conv_id, role="user", content=new_text))
    models.db.session.commit()

    history = [{"role": m.role, "content": m.content} for m in _ordered_messages(conv_id)]
    extra_blocks = []
    if data.get("web_search"):
        try:
            results = web_search_for_message(new_text)
            extra_blocks.append((
                "Web search results for grounding - answer EVERY part of the question "
                "above using the relevant results below; if a part genuinely isn't "
                "covered by any result, say so explicitly rather than skipping it. "
                "Cite the URLs you actually use, don't invent any",
                results,
            ))
        except Exception:
            traceback.print_exc(file=sys.stderr)
    try:
        knowledge = _project_knowledge_block(conv.project_id, new_text)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        knowledge = ""
    if knowledge:
        extra_blocks.append((
            "Reference material attached to this project - use it where relevant",
            knowledge,
        ))
    if extra_blocks:
        content = new_text
        for label, block in extra_blocks:
            content += f"\n\n[{label}]:\n{block}"
        history[-1] = {"role": "user", "content": content}

    reply_fn = deep_check_reply if data.get("deep_check") else agent.stream_reply

    def generate():
        yield f"data: {json.dumps({'truncated': True})}\n\n"
        chunks = []
        try:
            for delta in reply_fn(history, model, custom_instructions=custom_instructions):
                chunks.append(delta)
                yield f"data: {json.dumps({'delta': delta})}\n\n"
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            return
        full_reply = "".join(chunks).strip()
        models.db.session.add(models.Message(conversation_id=conv_id, role="assistant", content=full_reply))
        models.db.session.commit()
        html_out = render_message(full_reply)
        yield f"data: {json.dumps({'done': True, 'html': html_out})}\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


@app.route("/conv/<int:conv_id>/rename", methods=["POST"])
def conv_rename(conv_id):
    conv = models.Conversation.query.filter_by(id=conv_id, user_id=g.user.id).first()
    if not conv:
        return jsonify({"error": "Not found"}), 404
    data = request.get_json(silent=True) or {}
    title = (data.get("title") or "").strip()
    if not title:
        return jsonify({"error": "Title can't be empty"}), 400
    conv.title = title[:80]
    models.db.session.commit()
    return jsonify({"ok": True, "title": conv.title})


@app.route("/conv/<int:conv_id>/delete", methods=["POST"])
def conv_delete(conv_id):
    conv = models.Conversation.query.filter_by(id=conv_id, user_id=g.user.id).first()
    if conv:
        models.db.session.delete(conv)
        models.db.session.commit()
    return jsonify({"ok": True})


@app.route("/conv/<int:conv_id>/pin", methods=["POST"])
def conv_pin(conv_id):
    conv = models.Conversation.query.filter_by(id=conv_id, user_id=g.user.id).first()
    if not conv:
        return jsonify({"error": "Not found"}), 404
    conv.pinned = not conv.pinned
    models.db.session.commit()
    return jsonify({"ok": True, "pinned": conv.pinned})


@app.route("/conv/<int:conv_id>/project", methods=["POST"])
def conv_set_project(conv_id):
    conv = models.Conversation.query.filter_by(id=conv_id, user_id=g.user.id).first()
    if not conv:
        return jsonify({"error": "Not found"}), 404
    data = request.get_json(silent=True) or {}
    project_id = data.get("project_id") or None
    if project_id and not models.Project.query.filter_by(id=project_id, user_id=g.user.id).first():
        return jsonify({"error": "Project not found"}), 404
    conv.project_id = project_id
    models.db.session.commit()
    return jsonify({"ok": True})


@app.route("/projects/new", methods=["POST"])
def project_new():
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()[:60]
    if not name:
        return jsonify({"error": "Project name can't be empty"}), 400
    proj = models.Project(user_id=g.user.id, name=name)
    models.db.session.add(proj)
    models.db.session.commit()
    return jsonify({"ok": True, "id": proj.id, "name": proj.name})


@app.route("/projects/<int:project_id>/delete", methods=["POST"])
def project_delete(project_id):
    proj = models.Project.query.filter_by(id=project_id, user_id=g.user.id).first()
    if proj:
        models.Conversation.query.filter_by(project_id=project_id, user_id=g.user.id).update({"project_id": None})
        models.db.session.delete(proj)
        models.db.session.commit()
    return jsonify({"ok": True})


MAX_PROJECT_FILES = 10
MAX_PROJECT_FILE_CHARS = 40_000


@app.route("/projects/<int:project_id>/files", methods=["GET", "POST"])
def project_files(project_id):
    proj = models.Project.query.filter_by(id=project_id, user_id=g.user.id).first()
    if not proj:
        return jsonify({"error": "Not found"}), 404

    if request.method == "GET":
        return jsonify({
            "ok": True,
            "files": [{"id": f.id, "filename": f.filename, "chars": len(f.content)} for f in proj.files],
        })

    if len(proj.files) >= MAX_PROJECT_FILES:
        return jsonify({"error": f"This project already has the max of {MAX_PROJECT_FILES} files."}), 400
    data = request.get_json(silent=True) or {}
    filename = (data.get("filename") or "untitled.txt").strip()[:255]
    content = (data.get("content") or "")[:MAX_PROJECT_FILE_CHARS]
    if not content.strip():
        return jsonify({"error": "That file looks empty."}), 400

    pf = models.ProjectFile(project_id=proj.id, filename=filename, content=content)
    models.db.session.add(pf)
    models.db.session.commit()
    return jsonify({"ok": True, "id": pf.id, "filename": pf.filename, "chars": len(pf.content)})


@app.route("/projects/<int:project_id>/files/<int:file_id>/delete", methods=["POST"])
def project_file_delete(project_id, file_id):
    proj = models.Project.query.filter_by(id=project_id, user_id=g.user.id).first()
    if not proj:
        return jsonify({"error": "Not found"}), 404
    pf = models.ProjectFile.query.filter_by(id=file_id, project_id=proj.id).first()
    if pf:
        models.db.session.delete(pf)
        models.db.session.commit()
    return jsonify({"ok": True})


def _project_knowledge_block(project_id, query: str) -> str:
    """Retrieves the project's attached files, most-relevant-to-`query`
    chunks only (see project_rag.py) - falls back to including everything
    unchanged when the whole knowledge base is small enough that
    retrieval wouldn't change anything anyway."""
    if not project_id:
        return ""
    files = models.ProjectFile.query.filter_by(project_id=project_id).order_by(models.ProjectFile.id).all()
    if not files:
        return ""
    return project_rag.retrieve(query, [{"filename": f.filename, "content": f.content} for f in files])


@app.route("/conversations/clear", methods=["POST"])
def clear_all():
    conv_ids = [c.id for c in models.Conversation.query.filter_by(user_id=g.user.id).all()]
    _delete_conversations(conv_ids)
    models.Project.query.filter_by(user_id=g.user.id).delete()
    models.db.session.commit()
    return jsonify({"ok": True})


@app.route("/account/delete", methods=["POST"])
@limiter.limit("5 per hour")
def account_delete():
    user = g.user
    data = request.get_json(silent=True) or {}
    password = data.get("password") or ""
    if not user.check_password(password):
        return jsonify({"error": "Incorrect password."}), 400

    conv_ids = [c.id for c in models.Conversation.query.filter_by(user_id=user.id).all()]
    _delete_conversations(conv_ids)
    models.Project.query.filter_by(user_id=user.id).delete()

    try:
        shutil.rmtree(code_agent.user_workspace(user.id), ignore_errors=True)
    except Exception:
        traceback.print_exc(file=sys.stderr)

    models.db.session.delete(user)
    models.db.session.commit()
    session.pop("user_id", None)
    return jsonify({"ok": True})


@app.route("/artifacts", methods=["GET"])
def artifacts_gallery():
    user = g.user
    items = []
    convs = models.Conversation.query.filter_by(user_id=user.id).order_by(models.Conversation.id.desc()).all()
    for conv in convs:
        for m in conv.messages:
            if m.role != "assistant" or not m.content:
                continue
            _, arts = _extract_artifacts(m.content)
            for art in arts:
                items.append({
                    **art,
                    "highlighted": render_markdown(f"```{art['language']}\n{art['code']}\n```"),
                    "source_label": conv.title,
                    "source_url": ("/code" if conv.mode == "code" else "/") + f"?c={conv.id}",
                })

    return render_template(
        "artifacts.html",
        items=items,
        analyser_url=ANALYSER_URL,
        pygments_css=PYGMENTS_CSS,
        header_badge=HEADER_BADGE,
        user_email=user.email,
    )


def is_admin(user) -> bool:
    return bool(ADMIN_EMAILS) and user.email.lower() in ADMIN_EMAILS


@app.route("/admin", methods=["GET"])
def admin_dashboard():
    user = g.user
    if not is_admin(user):
        return jsonify({"error": "Not found"}), 404

    now = datetime.now(timezone.utc)
    week_ago = now - timedelta(days=7)

    total_users = models.User.query.count()
    total_conversations = models.Conversation.query.count()
    total_chat_convs = models.Conversation.query.filter_by(mode="chat").count()
    total_code_convs = models.Conversation.query.filter_by(mode="code").count()
    total_messages = models.Message.query.count()
    total_projects = models.Project.query.count()
    signups_7d = models.User.query.filter(models.User.created_at >= week_ago).count()

    recent_users = models.User.query.order_by(models.User.id.desc()).limit(15).all()
    # Per-user activity, cheapest as one query rather than N+1 inside the template.
    conv_counts = dict(
        models.db.session.query(models.Conversation.user_id, models.db.func.count(models.Conversation.id))
        .group_by(models.Conversation.user_id).all()
    )
    recent_rows = [
        {"email": u.email, "created_at": u.created_at, "conversations": conv_counts.get(u.id, 0)}
        for u in recent_users
    ]

    return render_template(
        "admin.html",
        user_email=user.email,
        analyser_url=ANALYSER_URL,
        header_badge=HEADER_BADGE,
        stats={
            "total_users": total_users,
            "total_conversations": total_conversations,
            "total_chat_convs": total_chat_convs,
            "total_code_convs": total_code_convs,
            "total_messages": total_messages,
            "total_projects": total_projects,
            "signups_7d": signups_7d,
        },
        recent_users=recent_rows,
        is_admin_view=True,
    )


# --- Code mode: a small Claude-Code-style agent (read/write files, run
# commands, gated on the user's approval) - see code_agent.py -------------
# Code sessions are just Conversations with mode="code" - they get the
# same sidebar (recent sessions, pin, projects, rename, delete) as chat
# mode for free. Tool-call/tool-result messages use Message's extra
# tool_calls_json / tool_call_id / tool_name columns instead of plain text.

def _pending(conv: models.Conversation):
    return json.loads(conv.pending_json) if conv.pending_json else None


def _save_pending(conv: models.Conversation, pending):
    conv.pending_json = json.dumps(pending) if pending else None
    models.db.session.commit()


def _add_tool_call_message(conv_id, call_id, name, args):
    m = models.Message(
        conversation_id=conv_id, role="assistant", content="",
        tool_calls_json=json.dumps([{"id": call_id, "type": "function", "function": {"name": name, "arguments": args}}]),
    )
    models.db.session.add(m)
    models.db.session.commit()


def _add_tool_result_message(conv_id, call_id, name, content):
    m = models.Message(conversation_id=conv_id, role="tool", content=content, tool_call_id=call_id, tool_name=name)
    models.db.session.add(m)
    models.db.session.commit()


def _api_messages_for(conv_id) -> list:
    """This conversation's messages, translated into the {"role","content",
    ["tool_calls"|"tool_call_id"/"name"]} shape agent.run_with_tools() expects."""
    rows = models.Message.query.filter_by(conversation_id=conv_id).order_by(models.Message.id).all()
    out = []
    for m in rows:
        if m.role == "assistant" and m.tool_calls_json:
            out.append({"role": "assistant", "content": None, "tool_calls": json.loads(m.tool_calls_json)})
        elif m.role == "tool":
            out.append({"role": "tool", "tool_call_id": m.tool_call_id, "name": m.tool_name, "content": m.content})
        else:
            out.append({"role": m.role, "content": m.content})
    return out


def _run_code_loop(conv_id: int, model: str, workspace_root, autonomous: bool = False):
    system = code_agent.SYSTEM_PROMPT_TEMPLATE.format(workspace=workspace_root)
    if autonomous:
        system += (
            "\n\nAutonomous mode is ON for this session: write_file and "
            "run_command execute immediately, without waiting for approval "
            "each time - the user is not watching every step live, so "
            "don't ask permission or narrate 'may I now...'. Just do the "
            "work, and give a clear summary of everything you did at the "
            "end. Still be careful: this is real, unreviewed execution."
        )

    last_call_signature = None
    repeat_count = 0

    for _ in range(code_agent.MAX_TOOL_STEPS):
        full_messages = [{"role": "system", "content": system}] + _api_messages_for(conv_id)
        try:
            text, tool_calls = agent.run_with_tools(full_messages, code_agent.TOOLS, model)
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            return

        if tool_calls:
            call = tool_calls[0]
            name, args, call_id = call["name"], call["arguments"], call["id"]

            # A smaller model can get stuck redoing the exact same action
            # (e.g. rewriting the same file repeatedly) instead of noticing
            # it's done and answering - burning through every remaining
            # step on nothing useful. Break out early instead of letting
            # that run all the way to the max-steps error.
            signature = (name, json.dumps(args, sort_keys=True))
            repeat_count = repeat_count + 1 if signature == last_call_signature else 0
            last_call_signature = signature
            if repeat_count >= 2:
                _add_tool_call_message(conv_id, call_id, name, args)
                _add_tool_result_message(
                    conv_id, call_id, name,
                    "Stopped: you called this exact action 3 times in a row without making "
                    "progress. Explain to the user what happened instead of repeating it again.",
                )
                yield f"data: {json.dumps({'action': {'kind': name, 'args': args, 'result': '(repeated action - stopped)'}})}\n\n"
                continue

            if name in code_agent.READ_ONLY_TOOLS:
                try:
                    result = code_agent.execute_tool(name, args, workspace_root)
                except Exception as e:
                    result = f"Error: {e}"
                _add_tool_call_message(conv_id, call_id, name, args)
                _add_tool_result_message(conv_id, call_id, name, result)
                yield f"data: {json.dumps({'action': {'kind': name, 'args': args, 'result': result[:1200]}})}\n\n"
                continue

            if name in code_agent.APPROVAL_TOOLS:
                if autonomous:
                    try:
                        result = code_agent.execute_pending(name, args, workspace_root)
                    except Exception as e:
                        result = f"Error: {e}"
                    _add_tool_call_message(conv_id, call_id, name, args)
                    _add_tool_result_message(conv_id, call_id, name, result)
                    yield f"data: {json.dumps({'action': {'kind': name, 'args': args, 'result': result[:1200], 'auto': True}})}\n\n"
                    continue
                try:
                    description = code_agent.describe_pending(name, args, workspace_root)
                except Exception as e:
                    description = {"kind": name, "error": str(e)}
                _add_tool_call_message(conv_id, call_id, name, args)
                conv = models.Conversation.query.get(conv_id)
                _save_pending(conv, {"id": call_id, "name": name, "arguments": args})
                yield f"data: {json.dumps({'pending': description})}\n\n"
                return

            # Unknown tool name - tell the model and keep going rather than crash the turn.
            _add_tool_call_message(conv_id, call_id, name, args)
            _add_tool_result_message(conv_id, call_id, name, f"Error: unknown tool '{name}'.")
            continue

        models.db.session.add(models.Message(conversation_id=conv_id, role="assistant", content=text or ""))
        models.db.session.commit()
        html_out = render_message(text or "")
        yield f"data: {json.dumps({'done': True, 'html': html_out})}\n\n"
        return

    yield f"data: {json.dumps({'error': 'Reached the max number of tool steps for this turn - try breaking the task down.'})}\n\n"


def _render_code_transcript(conv: models.Conversation):
    out = []
    messages = conv.messages
    i = 0
    while i < len(messages):
        m = messages[i]
        if m.role == "user":
            out.append({"kind": "user", "text": m.content})
        elif m.role == "assistant" and m.tool_calls_json:
            call = json.loads(m.tool_calls_json)[0]
            name = call["function"]["name"]
            args = call["function"]["arguments"]
            result = None
            if i + 1 < len(messages) and messages[i + 1].role == "tool":
                result = messages[i + 1].content
            out.append({"kind": "action", "name": name, "args": args, "result": result})
        elif m.role == "assistant" and m.content:
            out.append({"kind": "assistant", "html": render_message(m.content)})
        i += 1
    return out


@app.route("/code", methods=["GET"])
def code_home():
    user = g.user
    conv_id = request.args.get("c", type=int)
    conv = models.Conversation.query.filter_by(id=conv_id, user_id=user.id, mode="code").first() if conv_id else None

    workspace_root = code_agent.resolve_workspace(user.id, conv.workspace_path if conv else None)

    if conv:
        transcript = _render_code_transcript(conv)
        pending = _pending(conv)
        pending_desc = code_agent.describe_pending(pending["name"], pending["arguments"], workspace_root) if pending else None
    else:
        conv_id = None
        transcript = []
        pending_desc = None

    sidebar = _sidebar_data(user, "code", conv_id)
    return render_template(
        "code.html",
        model_choices=MODEL_CHOICES,
        current_model=agent.DEFAULT_MODEL,
        transcript=transcript,
        pending=pending_desc,
        active_id=conv_id,
        pinned=sidebar["pinned"],
        projects=sidebar["projects"],
        ungrouped=sidebar["ungrouped"],
        workspace=str(workspace_root),
        allow_custom_workspace=code_agent.ALLOW_CUSTOM_WORKSPACE,
        autonomous=conv.autonomous if conv else False,
        analyser_url=ANALYSER_URL,
        pygments_css=PYGMENTS_CSS,
        header_badge=HEADER_BADGE,
        user_email=user.email,
    )


@app.route("/code/<int:conv_id>/workspace", methods=["POST"])
def code_set_workspace(conv_id):
    if not code_agent.ALLOW_CUSTOM_WORKSPACE:
        return jsonify({"error": "Custom project folders aren't enabled on this deployment."}), 403
    conv = models.Conversation.query.filter_by(id=conv_id, user_id=g.user.id, mode="code").first()
    if not conv:
        return jsonify({"error": "Not found"}), 404
    data = request.get_json(silent=True) or {}
    try:
        path = code_agent.validate_custom_workspace(data.get("path"))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    conv.workspace_path = str(path)
    models.db.session.commit()
    return jsonify({"ok": True, "path": str(path)})


@app.route("/code/browse-folders", methods=["GET"])
def code_browse_folders():
    if not code_agent.ALLOW_CUSTOM_WORKSPACE:
        return jsonify({"error": "Custom project folders aren't enabled on this deployment."}), 403
    try:
        return jsonify({"ok": True, **code_agent.browse_folders(request.args.get("path"))})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/code/<int:conv_id>/autonomous", methods=["POST"])
def code_set_autonomous(conv_id):
    conv = models.Conversation.query.filter_by(id=conv_id, user_id=g.user.id, mode="code").first()
    if not conv:
        return jsonify({"error": "Not found"}), 404
    data = request.get_json(silent=True) or {}
    conv.autonomous = bool(data.get("autonomous"))
    models.db.session.commit()
    return jsonify({"ok": True, "autonomous": conv.autonomous})


@app.route("/code/send", methods=["POST"])
def code_send():
    user = g.user
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"error": "Please type a message."}), 400
    if len(message) > 8000:
        return jsonify({"error": "That message is too long (max 8000 characters)."}), 400

    model = pick_model(data, message)
    conv_id = data.get("conv_id")
    conv = models.Conversation.query.filter_by(id=conv_id, user_id=user.id, mode="code").first() if conv_id else None
    is_new = conv is None
    if is_new:
        conv = models.Conversation(user_id=user.id, mode="code", title=_make_title(message))
        requested_path = (data.get("workspace_path") or "").strip()
        if requested_path and code_agent.ALLOW_CUSTOM_WORKSPACE:
            try:
                conv.workspace_path = str(code_agent.validate_custom_workspace(requested_path))
            except ValueError as e:
                return jsonify({"error": str(e)}), 400
        conv.autonomous = bool(data.get("autonomous"))
        models.db.session.add(conv)
        models.db.session.commit()
    elif _pending(conv):
        return jsonify({"error": "Resolve the pending action first."}), 409

    models.db.session.add(models.Message(conversation_id=conv.id, role="user", content=message))
    models.db.session.commit()
    conv_id, conv_title, conv_autonomous = conv.id, conv.title, conv.autonomous
    workspace_root = code_agent.resolve_workspace(user.id, conv.workspace_path)

    def generate():
        if is_new:
            yield f"data: {json.dumps({'conv_id': conv_id, 'title': conv_title})}\n\n"
        yield from _run_code_loop(conv_id, model, workspace_root, autonomous=conv_autonomous)

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


@app.route("/code/approve", methods=["POST"])
def code_approve():
    data = request.get_json(silent=True) or {}
    conv = models.Conversation.query.filter_by(id=data.get("conv_id"), user_id=g.user.id, mode="code").first()
    if not conv:
        return jsonify({"error": "Not found"}), 404
    pending = _pending(conv)
    if not pending:
        return jsonify({"error": "Nothing pending."}), 400
    workspace_root = code_agent.resolve_workspace(g.user.id, conv.workspace_path)
    try:
        result = code_agent.execute_pending(pending["name"], pending["arguments"], workspace_root)
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        result = f"Error: {e}"
    _add_tool_result_message(conv.id, pending["id"], pending["name"], result)
    _save_pending(conv, None)
    conv_id = conv.id
    # "auto" has no fresh message to route on here (this just continues an
    # already-approved action) - stay on the accurate model rather than
    # guessing from nothing.
    model = ACCURATE_MODEL if data.get("model") == "auto" else pick_model(data)

    def generate():
        yield f"data: {json.dumps({'executed': result[:1200]})}\n\n"
        yield from _run_code_loop(conv_id, model, workspace_root, autonomous=conv.autonomous)

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


@app.route("/code/deny", methods=["POST"])
def code_deny():
    data = request.get_json(silent=True) or {}
    conv = models.Conversation.query.filter_by(id=data.get("conv_id"), user_id=g.user.id, mode="code").first()
    if not conv:
        return jsonify({"error": "Not found"}), 404
    pending = _pending(conv)
    if not pending:
        return jsonify({"error": "Nothing pending."}), 400
    _add_tool_result_message(
        conv.id, pending["id"], pending["name"],
        "The user denied this action. Do not attempt it again without asking first.",
    )
    _save_pending(conv, None)
    conv_id = conv.id
    model = ACCURATE_MODEL if data.get("model") == "auto" else pick_model(data)
    workspace_root = code_agent.resolve_workspace(g.user.id, conv.workspace_path)
    return Response(
        stream_with_context(_run_code_loop(conv_id, model, workspace_root, autonomous=conv.autonomous)),
        mimetype="text/event-stream",
    )


if __name__ == "__main__":
    # Local runs stay on 127.0.0.1 (private, this machine only). Defaults to
    # port 5001 so it can run alongside app.py (TXL Analyser, port 5000)
    # without a conflict. PORT takes priority (some launchers/hosts set it
    # to reassign the port automatically) - CHAT_PORT is the manual override.
    port = int(os.environ.get("PORT") or os.environ.get("CHAT_PORT", 5001))
    app.run(host="127.0.0.1", port=port, debug=False)
