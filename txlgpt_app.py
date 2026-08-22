"""
Txl GPT - Web UI
------------------------------------
A standalone ChatGPT-style web app: real accounts, persisted chat
history in a sidebar, streamed replies, image upload/vision, Work mode
(a tool-using agent), scheduled tasks, custom GPTs, and cross-chat
memory. Runs as its own process on its own port, with its own database
(txlgpt.db), its own accounts, its own workspace folder, and its own
connector modules (txlgpt_groq.py / txlgpt_ollama.py / txlgpt_gemini.py /
txlgpt_code_agent.py) - deliberately independent from app.py (TXL
Analyser) and chat_app.py (TXL Cloud). No shared code, no shared
runtime state, no shared identity - a genuinely separate product that
happens to live in the same repo.

Two interchangeable reply backends, picked with TXLGPT_BACKEND:
  - groq (default)  txlgpt_groq.py   - Groq's free cloud API. Fast, but
                                        capped at a daily free-tier
                                        token quota.
  - ollama           txlgpt_ollama.py - runs entirely on this machine
                                         via Ollama. No cap, nothing
                                         ever leaves this PC, but needs
                                         Ollama installed and a model
                                         pulled first.

Run:
    python txlgpt_app.py                                        # Groq, port 5003
    TXLGPT_BACKEND=ollama TXLGPT_PORT=5004 python txlgpt_app.py  # Ollama, port 5004
(PowerShell: $env:TXLGPT_BACKEND='ollama'; $env:TXLGPT_PORT='5004'; python txlgpt_app.py)
"""

import json
import os
import re
import secrets
import sys
import threading
import time
import traceback
from datetime import datetime, timedelta, timezone

import markdown as md
from flask import Flask, Response, g, jsonify, redirect, render_template, request, session, stream_with_context, url_for
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from pygments.formatters import HtmlFormatter

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import mailer
import txlgpt_code_agent as code_agent
import txlgpt_gemini as gemini
import txlgpt_models as models
from pdf_research_agent_local import web_search

app = Flask(__name__)
app.secret_key = os.environ.get("TXLGPT_SECRET_KEY") or os.environ.get("SECRET_KEY") or "dev-only-insecure-key-set-TXLGPT_SECRET_KEY-in-production"
# Always pick up template edits without a process restart - this only
# affects Jinja's file-watch behavior (dev convenience), not a security or
# performance concern in production (which shouldn't run this dev server
# at all - see the "WARNING: This is a development server" banner below).
app.config["TEMPLATES_AUTO_RELOAD"] = True
models.init_db(app)

limiter = Limiter(get_remote_address, app=app, default_limits=[], storage_uri="memory://")

BACKEND = os.environ.get("TXLGPT_BACKEND", "groq").strip().lower()

if BACKEND == "ollama":
    import txlgpt_ollama as agent
    FAST_MODEL = "qwen2.5:7b"
    ACCURATE_MODEL = "qwen2.5:14b"
    MODEL_CHOICES = [
        ("auto", "Auto"),
        (FAST_MODEL, "Balanced"),
        (ACCURATE_MODEL, "Accurate"),
    ]
    HEADER_BADGE = "Local · Unlimited"
else:
    import txlgpt_groq as agent
    ACCURATE_MODEL = "openai/gpt-oss-120b"
    FAST_MODEL = "openai/gpt-oss-20b"
    MODEL_CHOICES = [
        ("auto", "Auto"),
        (ACCURATE_MODEL, "Balanced"),
        (FAST_MODEL, "Fast"),
    ]
    HEADER_BADGE = "Free · Groq-powered"

VALID_MODELS = {m for m, _ in MODEL_CHOICES}

SYSTEM_PROMPT = """You are Txl GPT, a free, helpful AI chat assistant. \
You are your own independent product, built to run on a free open-weight \
model rather than OpenAI's ChatGPT or Anthropic's Claude - if asked, be \
upfront about that rather than claiming to be either of them. Be clear and \
warm. Prioritize being accurate and substantive over sounding confident - \
give real, specific, correct answers with the actual details/numbers/names \
requested; if you're not sure of something, say so plainly rather than \
guessing or filling space with vague, generic-sounding filler. Don't hedge \
on things you do know just to seem cautious.

Match your depth to the request. A quick factual question ("what's 12% of \
80", "what year did X happen") gets a short, direct answer - don't pad it \
with headers or bullet after-thoughts it doesn't need. But when asked to \
explain, analyze, describe, or walk through something substantial - an \
image, a document, a comparison, a concept - go deep: don't just transcribe \
or list what's there, explain what each part actually means and why it \
matters, the way a knowledgeable person would walk a colleague through it. \
Use headings and bullets to structure a longer explanation, and draw on \
what you actually know about the subject (not just what's visible) to add \
real context per point - a name alone ("Feature X") is worth far less than \
a sentence on what Feature X does and why someone would care. Thin, \
list-only answers to substantial questions are a failure mode - elaborate. \
Use markdown (headings, lists, code blocks) when it genuinely helps \
readability. You're also a capable coding assistant: when asked for code, \
write correct, working code and always put it in a fenced code block \
tagged with the right language (e.g. ```python) so it can be \
syntax-highlighted - explain briefly around it, but don't pad with \
unnecessary commentary."""

_CODE_HINT_RE = re.compile(
    r"```|\bdef \b|\bfunction\b|\bclass \b|\bimport \b|SELECT .* FROM|\berror\b|\btraceback\b|\bdebug\b|\bfix\b",
    re.IGNORECASE,
)


def route_model(message: str) -> str:
    """Heuristic router for the 'Auto' model choice: short, simple-looking
    questions go to the fast model; anything long or code/debugging-shaped
    goes to the bigger/more accurate one. Pure text heuristics - adds no
    extra API call or latency."""
    text = message or ""
    looks_complex = len(text.split()) > 40 or text.count("?") > 1 or bool(_CODE_HINT_RE.search(text))
    return ACCURATE_MODEL if looks_complex else FAST_MODEL


def pick_model(data, message: str = "") -> str:
    model = data.get("model", "")
    if model not in VALID_MODELS:
        return agent.DEFAULT_MODEL
    if model == "auto":
        return route_model(message)
    return model


PYGMENTS_CSS = (
    HtmlFormatter(style="xcode", nobackground=True).get_style_defs(".bubble .codehilite")
    + "\n@media (prefers-color-scheme: dark) {\n"
    + HtmlFormatter(style="monokai", nobackground=True).get_style_defs(".bubble .codehilite")
    + "\n}"
)

MD_EXTENSIONS = ["extra", "sane_lists", "codehilite"]
MD_EXTENSION_CONFIGS = {"codehilite": {"guess_lang": False}}


def render_message(text: str) -> str:
    return md.markdown(text, extensions=MD_EXTENSIONS, extension_configs=MD_EXTENSION_CONFIGS)


_IMAGE_DATA_URL_RE = re.compile(r"^data:image/(png|jpeg|jpg|webp|gif);base64,[A-Za-z0-9+/]+=*$")
MAX_IMAGE_B64_CHARS = 8_000_000  # ~6 MB raw image, base64-inflated


class InvalidImage(ValueError):
    pass


def _validate_image(image_data_url):
    if not image_data_url:
        return None
    if len(image_data_url) > MAX_IMAGE_B64_CHARS:
        raise InvalidImage("That image is too large (max ~6 MB).")
    if not _IMAGE_DATA_URL_RE.match(image_data_url):
        raise InvalidImage("Unsupported image data.")
    return image_data_url


def _make_title(message: str) -> str:
    title = " ".join(message.split())
    return title if len(title) <= 48 else title[:47].rstrip() + "…"


def web_search_for_message(message: str) -> str:
    """Deterministic web search for the 'Web Search' plugin toggle - not
    left up to the model to decide whether to search (see
    pdf_research_agent_local.py for why)."""
    try:
        return web_search(message, max_results=5)
    except Exception as e:
        return f"Search failed: {e}"


# --- Deep Research: like Web Search, but splits the question into several
# angles and searches each with more results, for a longer, structured,
# multi-source report instead of a single grounded paragraph. ---------------

_COMPOUND_JOINER_RE = re.compile(
    r",?\s+and\s+(what|when|where|who|why|how|which|is|are|does|do|can|will)\b", re.IGNORECASE
)
MAX_RESEARCH_SUBQUERIES = 4


def _split_research_queries(message: str) -> list:
    parts = [p.strip() for p in message.split("?") if p.strip()]
    if len(parts) > 1:
        return [p + "?" for p in parts][:MAX_RESEARCH_SUBQUERIES]
    m = _COMPOUND_JOINER_RE.search(message)
    if m:
        first = message[:m.start()].strip()
        second = (m.group(1) + message[m.end():]).strip()
        candidates = [q for q in (first, second) if len(q) > 3]
        if len(candidates) > 1:
            return candidates
    return [message]


def deep_research_for_message(message: str) -> str:
    """Multi-query web research for the 'Deep Research' composer toggle -
    more sources per query and a report-style instruction, vs. Web
    Search's single quick lookup."""
    queries = _split_research_queries(message)
    blocks = []
    for q in queries:
        try:
            result = web_search(q, max_results=8)
        except Exception as e:
            result = f"Search failed: {e}"
        blocks.append(f'Results for "{q}":\n{result}')
    return "\n\n".join(blocks)


# --- Memory: facts Txl GPT remembers about this user across every chat -----

MAX_MEMORIES_IN_PROMPT = 30
_REMEMBER_RE = re.compile(r"^\s*remember\s+(?:that\s+)?(.+)$", re.IGNORECASE | re.DOTALL)


def _memory_block(user_id: int) -> str:
    memories = models.Memory.query.filter_by(user_id=user_id).order_by(models.Memory.id.desc()).limit(MAX_MEMORIES_IN_PROMPT).all()
    if not memories:
        return ""
    facts = "\n".join(f"- {m.content}" for m in reversed(memories))
    return (
        "\n\nThings you've been asked to remember about this user, from "
        f"past conversations - use them where relevant:\n{facts}"
    )


def _maybe_capture_memory(user_id: int, message: str) -> None:
    """Lightweight auto-capture: 'remember (that) X' saves X as a standing
    memory fact, no extra model call needed. Doesn't stop the message from
    also being answered normally - see chat_send."""
    m = _REMEMBER_RE.match(message)
    if not m:
        return
    fact = " ".join(m.group(1).split())
    if fact and len(fact) <= 2000:
        models.db.session.add(models.Memory(user_id=user_id, content=fact))
        models.db.session.commit()


def _effective_system_prompt(conv, user_id: int) -> str:
    """The base persona for this conversation - a custom GPT's own
    instructions if it was started under one, otherwise Txl GPT's default -
    plus this user's remembered facts, folded in on every turn."""
    if conv.gpt_id:
        gpt = models.CustomGPT.query.filter_by(id=conv.gpt_id, user_id=user_id).first()
        if gpt:
            base = (
                f"You are \"{gpt.name}\", a custom version of Txl GPT with these standing "
                f"instructions:\n{gpt.instructions}\n\nStay in this persona unless it "
                "conflicts with being safe or honest."
            )
            return base + _memory_block(user_id)
    return SYSTEM_PROMPT + _memory_block(user_id)


def _valid_email(email: str) -> bool:
    return "@" in email and "." in email.split("@")[-1] and len(email) <= 255


# Optional extra password gate on TOP of accounts - set SITE_PASSWORD in the
# environment (e.g. on Render) to require a shared password before anyone
# can even reach the login page on a public deployment. Unset = no gate.
SITE_PASSWORD = os.environ.get("SITE_PASSWORD")

# Optional: require an invite code to sign up (on top of SITE_PASSWORD, if
# also set) - stops anyone who has SITE_PASSWORD from creating their own
# account and using Work mode's run_command on this server. Unset = signup
# stays open to anyone who reaches it, same as before.
SIGNUP_INVITE_CODE = os.environ.get("SIGNUP_INVITE_CODE")

PUBLIC_ENDPOINTS = {"login", "signup", "static", "forgot_password", "reset_password"}


@app.before_request
def require_password():
    if not SITE_PASSWORD:
        return None
    auth = request.authorization
    if not auth or auth.password != SITE_PASSWORD:
        return Response(
            "Login required", 401, {"WWW-Authenticate": 'Basic realm="Txl GPT"'}
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
    traceback.print_exc(file=sys.stderr)
    if request.is_json:
        return jsonify({"error": "Something went wrong on our end. Please try again."}), 500
    return Response("<h1>Something went wrong</h1><p>Please try again.</p>", status=500, mimetype="text/html")


# --- Auth --------------------------------------------------------------

@app.route("/signup", methods=["GET", "POST"])
@limiter.limit("8 per hour", methods=["POST"])
def signup():
    if request.method == "GET":
        return render_template("txlgpt_signup.html", error=None, email="", invite_required=bool(SIGNUP_INVITE_CODE))
    email = (request.form.get("email") or "").strip().lower()
    password = request.form.get("password") or ""
    confirm = request.form.get("confirm") or ""
    invite_code = request.form.get("invite_code") or ""
    invite_required = bool(SIGNUP_INVITE_CODE)
    if invite_required and invite_code != SIGNUP_INVITE_CODE:
        return render_template("txlgpt_signup.html", error="Incorrect invite code.", email=email, invite_required=True)
    if not _valid_email(email):
        return render_template("txlgpt_signup.html", error="Enter a valid email address.", email=email, invite_required=invite_required)
    if len(password) < 8:
        return render_template("txlgpt_signup.html", error="Password must be at least 8 characters.", email=email, invite_required=invite_required)
    if password != confirm:
        return render_template("txlgpt_signup.html", error="Passwords don't match.", email=email, invite_required=invite_required)
    if models.User.query.filter_by(email=email).first():
        return render_template("txlgpt_signup.html", error="An account with that email already exists.", email=email, invite_required=invite_required)

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
        return render_template("txlgpt_login.html", error=None, email="")
    email = (request.form.get("email") or "").strip().lower()
    password = request.form.get("password") or ""
    user = models.User.query.filter_by(email=email).first()
    if not user or not user.check_password(password):
        return render_template("txlgpt_login.html", error="Incorrect email or password.", email=email)
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
        return render_template("txlgpt_forgot_password.html", error=None, sent=False, mail_configured=mailer.is_configured())

    if not mailer.is_configured():
        return render_template(
            "txlgpt_forgot_password.html", sent=False, mail_configured=False,
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
            mailer.send_password_reset_email(user.email, reset_url, product_name="Txl GPT")
        except Exception:
            traceback.print_exc(file=sys.stderr)
            # Don't leak send failures to the client - same generic message either way.
    return render_template("txlgpt_forgot_password.html", error=None, sent=True, mail_configured=True)


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
        return render_template("txlgpt_reset_password.html", token=token, valid=valid, error=None)

    if not valid:
        return render_template("txlgpt_reset_password.html", token=token, valid=False, error=None)

    password = request.form.get("password") or ""
    confirm = request.form.get("confirm") or ""
    if len(password) < 8:
        return render_template("txlgpt_reset_password.html", token=token, valid=True, error="Password must be at least 8 characters.")
    if password != confirm:
        return render_template("txlgpt_reset_password.html", token=token, valid=True, error="Passwords don't match.")

    user.set_password(password)
    user.reset_token = None
    user.reset_token_expires = None
    models.db.session.commit()
    return redirect(url_for("login"))


# --- Chat ----------------------------------------------------------------

def _sidebar_data(user, mode: str, active_id=None) -> dict:
    """Groups this user's conversations OF ONE MODE ("chat" or "work") for
    the sidebar: pinned chats (regardless of project) first, then each
    project's own (non-pinned, same-mode) chats, then everything else
    ungrouped. Most-recently-created first. Projects are shared across
    both modes - only the chat lists shown under them are filtered."""
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

    active_gpt = None
    if conv:
        msgs = conv.messages
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
        if conv.gpt_id:
            active_gpt = models.CustomGPT.query.filter_by(id=conv.gpt_id, user_id=user.id).first()
    else:
        conv_id = None
        history = []
        gpt_id = request.args.get("gpt", type=int)
        if gpt_id:
            active_gpt = models.CustomGPT.query.filter_by(id=gpt_id, user_id=user.id).first()

    sidebar = _sidebar_data(user, "chat", conv_id)
    gpts = models.CustomGPT.query.filter_by(user_id=user.id).order_by(models.CustomGPT.id.desc()).all()
    return render_template(
        "txlgpt_chat.html",
        model_choices=MODEL_CHOICES,
        history=history,
        active_id=conv_id,
        active_gpt=active_gpt,
        pinned=sidebar["pinned"],
        projects=sidebar["projects"],
        ungrouped=sidebar["ungrouped"],
        gpts=gpts,
        pygments_css=PYGMENTS_CSS,
        header_badge=HEADER_BADGE,
        user_email=user.email,
        image_gen_available=gemini.is_configured(),
        thinking_available=(BACKEND != "ollama"),
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
    thinking = bool(data.get("thinking")) and BACKEND != "ollama"

    conv_id = data.get("conv_id")
    conv = models.Conversation.query.filter_by(id=conv_id, user_id=user.id, mode="chat").first() if conv_id else None
    is_new = conv is None
    if is_new:
        conv = models.Conversation(user_id=user.id, mode="chat", title=_make_title(message))
        gpt_id = data.get("gpt_id")
        if gpt_id and models.CustomGPT.query.filter_by(id=gpt_id, user_id=user.id).first():
            conv.gpt_id = gpt_id
        models.db.session.add(conv)
        models.db.session.commit()

    _maybe_capture_memory(user.id, message)

    user_msg = models.Message(conversation_id=conv.id, role="user", content=message, image_data=image_data_url)
    models.db.session.add(user_msg)
    models.db.session.commit()
    history = [{"role": m.role, "content": m.content} for m in conv.messages]
    conv_id, conv_title = conv.id, conv.title
    system_prompt = _effective_system_prompt(conv, user.id)

    # "Deep Research" toggle takes priority over the simpler "Web Search"
    # toggle (more sources, more sub-queries) - fold results into the API
    # call only, the clean original message is what's stored above.
    if history and history[-1]["role"] == "user":
        if data.get("deep_research"):
            try:
                results = deep_research_for_message(message)
                history[-1] = {**history[-1], "content": (
                    history[-1]["content"]
                    + "\n\n[Deep research - web search results across several angles of "
                    "this question. Write a thorough, well-structured report (use headings "
                    "for distinct sub-topics) that answers every part of the question, citing "
                    "the URLs you actually use - don't invent any:]\n" + results
                )}
            except Exception:
                traceback.print_exc(file=sys.stderr)
        elif data.get("web_search"):
            try:
                results = web_search_for_message(message)
                history[-1] = {**history[-1], "content": (
                    history[-1]["content"]
                    + "\n\n[Web search results for grounding - answer using the "
                    "relevant results below; cite the URLs you actually use, "
                    "don't invent any:]\n" + results
                )}
            except Exception:
                traceback.print_exc(file=sys.stderr)

    def generate():
        if is_new:
            yield f"data: {json.dumps({'conv_id': conv_id, 'title': conv_title})}\n\n"
        chunks = []
        try:
            for delta in agent.stream_reply(
                history, model, image_data_url=image_data_url, system_prompt=system_prompt,
                reasoning_effort="high" if thinking else None,
            ):
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


@app.route("/conv/<int:conv_id>/regenerate", methods=["POST"])
def chat_regenerate(conv_id):
    user = g.user
    conv = models.Conversation.query.filter_by(id=conv_id, user_id=user.id, mode="chat").first()
    if not conv or not conv.messages or conv.messages[-1].role != "assistant":
        return jsonify({"error": "Nothing to regenerate."}), 400

    old_assistant_id = conv.messages[-1].id
    history = [{"role": m.role, "content": m.content} for m in conv.messages[:-1]]
    data = request.get_json(silent=True) or {}
    model = pick_model(data, history[-1]["content"] if history else "")
    thinking = bool(data.get("thinking")) and BACKEND != "ollama"
    system_prompt = _effective_system_prompt(conv, user.id)

    def generate():
        chunks = []
        try:
            for delta in agent.stream_reply(
                history, model, system_prompt=system_prompt, reasoning_effort="high" if thinking else None,
            ):
                chunks.append(delta)
                yield f"data: {json.dumps({'delta': delta})}\n\n"
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            return
        full_reply = "".join(chunks).strip()
        old = models.Message.query.get(old_assistant_id)
        if old:
            models.db.session.delete(old)
        models.db.session.add(models.Message(conversation_id=conv_id, role="assistant", content=full_reply))
        models.db.session.commit()
        html_out = render_message(full_reply)
        yield f"data: {json.dumps({'done': True, 'html': html_out})}\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


@app.route("/conv/<int:conv_id>/rename", methods=["POST"])
def chat_rename(conv_id):
    user = g.user
    conv = models.Conversation.query.filter_by(id=conv_id, user_id=user.id).first()
    if not conv:
        return jsonify({"error": "Not found."}), 404
    title = (request.get_json(silent=True) or {}).get("title", "").strip()
    if not title:
        return jsonify({"error": "Title can't be empty."}), 400
    conv.title = title[:200]
    models.db.session.commit()
    return jsonify({"ok": True, "title": conv.title})


@app.route("/conv/<int:conv_id>/delete", methods=["POST"])
def chat_delete(conv_id):
    user = g.user
    conv = models.Conversation.query.filter_by(id=conv_id, user_id=user.id).first()
    if conv:
        models.db.session.delete(conv)
        models.db.session.commit()
    return jsonify({"ok": True})


@app.route("/conv/<int:conv_id>/pin", methods=["POST"])
def chat_pin(conv_id):
    user = g.user
    conv = models.Conversation.query.filter_by(id=conv_id, user_id=user.id).first()
    if not conv:
        return jsonify({"error": "Not found."}), 404
    conv.pinned = not conv.pinned
    models.db.session.commit()
    return jsonify({"ok": True, "pinned": conv.pinned})


@app.route("/conv/<int:conv_id>/project", methods=["POST"])
def chat_set_project(conv_id):
    user = g.user
    conv = models.Conversation.query.filter_by(id=conv_id, user_id=user.id).first()
    if not conv:
        return jsonify({"error": "Not found."}), 404
    project_id = (request.get_json(silent=True) or {}).get("project_id")
    if project_id:
        project = models.Project.query.filter_by(id=project_id, user_id=user.id).first()
        if not project:
            return jsonify({"error": "Project not found."}), 404
        conv.project_id = project.id
    else:
        conv.project_id = None
    models.db.session.commit()
    return jsonify({"ok": True})


# --- Projects --------------------------------------------------------------

@app.route("/projects/new", methods=["POST"])
def projects_new():
    user = g.user
    name = (request.get_json(silent=True) or {}).get("name", "").strip()
    if not name:
        return jsonify({"error": "Please enter a project name."}), 400
    project = models.Project(user_id=user.id, name=name[:120])
    models.db.session.add(project)
    models.db.session.commit()
    return jsonify({"ok": True, "id": project.id, "name": project.name})


@app.route("/projects/<int:project_id>/delete", methods=["POST"])
def projects_delete(project_id):
    user = g.user
    project = models.Project.query.filter_by(id=project_id, user_id=user.id).first()
    if project:
        models.Conversation.query.filter_by(project_id=project.id, user_id=user.id).update({"project_id": None})
        models.db.session.delete(project)
        models.db.session.commit()
    return jsonify({"ok": True})


# --- Memory ----------------------------------------------------------------

@app.route("/memory", methods=["GET"])
def memory_page():
    user = g.user
    memories = models.Memory.query.filter_by(user_id=user.id).order_by(models.Memory.id.desc()).all()
    return render_template("txlgpt_memory.html", memories=memories, user_email=user.email, header_badge=HEADER_BADGE)


@app.route("/memory/new", methods=["POST"])
def memory_new():
    user = g.user
    content = (request.get_json(silent=True) or {}).get("content", "").strip()
    if not content:
        return jsonify({"error": "Please enter something to remember."}), 400
    if len(content) > 2000:
        return jsonify({"error": "That's too long (max 2000 characters)."}), 400
    memory = models.Memory(user_id=user.id, content=content)
    models.db.session.add(memory)
    models.db.session.commit()
    return jsonify({"ok": True, "id": memory.id})


@app.route("/memory/<int:memory_id>/delete", methods=["POST"])
def memory_delete(memory_id):
    user = g.user
    memory = models.Memory.query.filter_by(id=memory_id, user_id=user.id).first()
    if memory:
        models.db.session.delete(memory)
        models.db.session.commit()
    return jsonify({"ok": True})


# --- Custom GPTs -------------------------------------------------------------

@app.route("/gpts", methods=["GET"])
def gpts_page():
    user = g.user
    gpts = models.CustomGPT.query.filter_by(user_id=user.id).order_by(models.CustomGPT.id.desc()).all()
    return render_template("txlgpt_gpts.html", gpts=gpts, user_email=user.email, header_badge=HEADER_BADGE)


@app.route("/gpts/new", methods=["POST"])
def gpts_new():
    user = g.user
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    instructions = (data.get("instructions") or "").strip()
    description = (data.get("description") or "").strip()
    icon = (data.get("icon") or "🤖").strip()[:8] or "🤖"
    if not name:
        return jsonify({"error": "Please enter a name."}), 400
    if not instructions:
        return jsonify({"error": "Please enter instructions."}), 400
    if len(instructions) > 6000:
        return jsonify({"error": "Instructions are too long (max 6000 characters)."}), 400
    gpt = models.CustomGPT(user_id=user.id, name=name[:80], description=description[:200] or None, instructions=instructions, icon=icon)
    models.db.session.add(gpt)
    models.db.session.commit()
    return jsonify({"ok": True, "id": gpt.id})


@app.route("/gpts/<int:gpt_id>/delete", methods=["POST"])
def gpts_delete(gpt_id):
    user = g.user
    gpt = models.CustomGPT.query.filter_by(id=gpt_id, user_id=user.id).first()
    if gpt:
        models.Conversation.query.filter_by(gpt_id=gpt.id, user_id=user.id).update({"gpt_id": None})
        models.db.session.delete(gpt)
        models.db.session.commit()
    return jsonify({"ok": True})


# --- Image generation --------------------------------------------------------

@app.route("/generate-image", methods=["POST"])
def generate_image_route():
    user = g.user
    data = request.get_json(silent=True) or {}
    prompt = (data.get("prompt") or "").strip()
    thinking = bool(data.get("thinking"))
    if not prompt:
        return jsonify({"error": "Please describe the image you want."}), 400
    if len(prompt) > 2000:
        return jsonify({"error": "That prompt is too long (max 2000 characters)."}), 400
    if not gemini.is_configured():
        return jsonify({"error": "Image generation isn't set up on this deployment yet - ask whoever runs it to configure GEMINI_API_KEY."}), 400

    conv_id = data.get("conv_id")
    conv = models.Conversation.query.filter_by(id=conv_id, user_id=user.id, mode="chat").first() if conv_id else None
    is_new = conv is None
    if is_new:
        conv = models.Conversation(user_id=user.id, mode="chat", title=_make_title(prompt))
        models.db.session.add(conv)
        models.db.session.commit()

    models.db.session.add(models.Message(conversation_id=conv.id, role="user", content=prompt))
    models.db.session.commit()

    used_prompt = prompt
    if thinking:
        used_prompt = gemini.expand_image_prompt(prompt)

    try:
        image_data_url = gemini.generate_image(used_prompt)
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        return jsonify({"error": str(e)}), 500

    caption = "Here's the image I generated:" if not thinking else f"Here's the image I generated (expanded prompt: _{used_prompt}_):"
    assistant_msg = models.Message(
        conversation_id=conv.id, role="assistant", content=caption, image_data=image_data_url,
    )
    models.db.session.add(assistant_msg)
    models.db.session.commit()

    return jsonify({
        "ok": True, "conv_id": conv.id, "title": conv.title, "is_new": is_new,
        "image_data": image_data_url, "html": render_message(caption),
    })


# --- Scheduled tasks -----------------------------------------------------
# A prompt the user wants run automatically later, once or on a recurring
# schedule. Runs in-process via a background thread poller (_scheduler_loop)
# that checks for due tasks every 30s - so tasks only fire while this
# process itself is running, same as any other local dev app.

def _compute_next_run(schedule_type: str, run_at_time: str = None, interval_minutes: int = None, base: datetime = None) -> datetime:
    base = base or datetime.now(timezone.utc)
    if schedule_type == "interval":
        return base + timedelta(minutes=interval_minutes)
    if schedule_type == "daily":
        hh, mm = (int(x) for x in run_at_time.split(":"))
        candidate = base.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if candidate <= base:
            candidate += timedelta(days=1)
        return candidate
    return base  # "once" - caller sets next_run directly from the user's chosen date/time


@app.route("/scheduled", methods=["GET"])
def scheduled_page():
    user = g.user
    tasks = models.ScheduledTask.query.filter_by(user_id=user.id).order_by(models.ScheduledTask.next_run.asc()).all()
    return render_template("txlgpt_scheduled.html", tasks=tasks, user_email=user.email, header_badge=HEADER_BADGE)


@app.route("/scheduled/new", methods=["POST"])
def scheduled_new():
    user = g.user
    data = request.get_json(silent=True) or {}
    prompt = (data.get("prompt") or "").strip()
    schedule_type = data.get("schedule_type")
    if not prompt:
        return jsonify({"error": "Please enter a prompt."}), 400
    if len(prompt) > 4000:
        return jsonify({"error": "That prompt is too long (max 4000 characters)."}), 400
    if schedule_type not in ("once", "daily", "interval"):
        return jsonify({"error": "Invalid schedule type."}), 400

    run_at_time = None
    interval_minutes = None
    if schedule_type == "once":
        try:
            next_run = datetime.fromisoformat((data.get("run_at") or "").replace("Z", "+00:00"))
            if next_run.tzinfo is None:
                next_run = next_run.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            return jsonify({"error": "Please choose a valid date and time."}), 400
        if next_run <= datetime.now(timezone.utc):
            return jsonify({"error": "Pick a time in the future."}), 400
    elif schedule_type == "daily":
        run_at_time = data.get("run_at_time") or ""
        if not re.match(r"^\d{2}:\d{2}$", run_at_time):
            return jsonify({"error": "Please choose a valid time."}), 400
        next_run = _compute_next_run("daily", run_at_time=run_at_time)
    else:  # interval
        try:
            interval_minutes = int(data.get("interval_minutes"))
        except (TypeError, ValueError):
            return jsonify({"error": "Please enter a valid number of minutes."}), 400
        if interval_minutes < 5:
            return jsonify({"error": "Minimum interval is 5 minutes."}), 400
        next_run = _compute_next_run("interval", interval_minutes=interval_minutes)

    task = models.ScheduledTask(
        user_id=user.id, prompt=prompt, schedule_type=schedule_type,
        run_at_time=run_at_time, interval_minutes=interval_minutes, next_run=next_run,
    )
    models.db.session.add(task)
    models.db.session.commit()
    return jsonify({"ok": True, "id": task.id})


@app.route("/scheduled/<int:task_id>/toggle", methods=["POST"])
def scheduled_toggle(task_id):
    user = g.user
    task = models.ScheduledTask.query.filter_by(id=task_id, user_id=user.id).first()
    if not task:
        return jsonify({"error": "Not found."}), 404
    task.enabled = not task.enabled
    models.db.session.commit()
    return jsonify({"ok": True, "enabled": task.enabled})


@app.route("/scheduled/<int:task_id>/delete", methods=["POST"])
def scheduled_delete(task_id):
    user = g.user
    task = models.ScheduledTask.query.filter_by(id=task_id, user_id=user.id).first()
    if task:
        models.db.session.delete(task)
        models.db.session.commit()
    return jsonify({"ok": True})


def _run_scheduled_task(task_id: int) -> None:
    with app.app_context():
        task = models.ScheduledTask.query.get(task_id)
        if not task or not task.enabled:
            return

        conv = models.Conversation.query.filter_by(id=task.conversation_id, user_id=task.user_id).first() if task.conversation_id else None
        if not conv:
            conv = models.Conversation(user_id=task.user_id, title="⏰ " + _make_title(task.prompt))
            models.db.session.add(conv)
            models.db.session.commit()
            task.conversation_id = conv.id

        models.db.session.add(models.Message(conversation_id=conv.id, role="user", content=task.prompt))
        models.db.session.commit()
        history = [{"role": m.role, "content": m.content} for m in conv.messages]

        try:
            full_reply = "".join(agent.stream_reply(history, agent.DEFAULT_MODEL, system_prompt=SYSTEM_PROMPT)).strip()
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            full_reply = f"(This scheduled run failed: {e})"
        models.db.session.add(models.Message(conversation_id=conv.id, role="assistant", content=full_reply))

        task.last_run = datetime.now(timezone.utc)
        if task.schedule_type == "once":
            task.enabled = False
        else:
            task.next_run = _compute_next_run(
                task.schedule_type, run_at_time=task.run_at_time,
                interval_minutes=task.interval_minutes, base=task.last_run,
            )
        models.db.session.commit()


def _scheduler_loop() -> None:
    while True:
        time.sleep(30)
        try:
            with app.app_context():
                now = datetime.now(timezone.utc)
                due_ids = [
                    t.id for t in models.ScheduledTask.query.filter(
                        models.ScheduledTask.enabled.is_(True), models.ScheduledTask.next_run <= now,
                    ).all()
                ]
            for task_id in due_ids:
                _run_scheduled_task(task_id)
        except Exception:
            traceback.print_exc(file=sys.stderr)


# Started at import time (not inside `if __name__ == "__main__"`) so it
# also runs under a production WSGI server like gunicorn, which imports
# this module and calls `app` directly rather than executing it as a
# script - code gated behind `__main__` would never run there, silently
# breaking Scheduled tasks on a real deployment. The WERKZEUG_RUN_MAIN
# check only matters for local `python txlgpt_app.py` runs with
# FLASK_DEBUG=1: it stops the debug reloader's parent watcher process from
# starting a second, redundant copy of this thread.
if os.environ.get("FLASK_DEBUG") != "1" or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
    threading.Thread(target=_scheduler_loop, daemon=True).start()


# --- Work mode: a tool-using agent for multi-step tasks & coding -----------
# Uses txlgpt_code_agent.py, Txl GPT's own sandboxed tool-calling engine,
# plus one more read-only tool, web_search, so it's a genuine "research +
# files + commands" work agent, not just a coding agent. Read-only tools
# run automatically; write_file/run_command always stop for the user's
# explicit approval first.

_WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web and return a summary of the top results. Use this for anything you don't already know or that may have changed recently.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The search query."}},
            "required": ["query"],
        },
    },
}
WORK_TOOLS = code_agent.TOOLS + [_WEB_SEARCH_TOOL]
WORK_READ_ONLY_TOOLS = code_agent.READ_ONLY_TOOLS | {"web_search"}

WORK_SYSTEM_PROMPT_TEMPLATE = """You are Txl GPT in Work mode: an agent \
with tools to search the web, inspect and modify files, and run shell \
commands, inside the user's sandboxed workspace at {workspace}. You cannot \
see or touch anything outside that folder.

- Use web_search for anything you're not certain of or that may have \
changed recently - don't guess or make up facts, URLs, or numbers.
- Use list_directory and read_file to look around before making changes - \
don't guess at file contents.
- Use fetch_url to check whether something you built (or any URL, \
including localhost) is actually responding - don't guess what a running \
server returns.
- Use write_file to create or edit files, and run_command for terminal \
commands. Both require the user's explicit approval before they actually \
execute - after calling one, wait for the tool result to tell you whether \
it was approved or denied. Never assume it already happened, and never \
say a file was written or a command was run unless a tool result actually \
confirmed it.
- Be concise. Let the actions speak; don't narrate every step at length.
- If the user denies an action, don't retry the same thing - ask what \
they'd prefer instead.
- Break multi-step tasks down and work through them one tool call at a \
time rather than trying to plan everything in one message."""


def _pending(conv):
    return json.loads(conv.pending_json) if conv.pending_json else None


def _save_pending(conv, pending):
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


def _work_api_messages_for(conv_id) -> list:
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


def _execute_work_tool(name: str, args: dict, workspace_root) -> str:
    if name == "web_search":
        return web_search_for_message(args.get("query", ""))
    return code_agent.execute_tool(name, args, workspace_root)


def _render_work_transcript(conv):
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


def _run_work_loop(conv_id: int, model: str, workspace_root):
    system = WORK_SYSTEM_PROMPT_TEMPLATE.format(workspace=workspace_root)
    last_call_signature = None
    repeat_count = 0

    for _ in range(code_agent.MAX_TOOL_STEPS):
        full_messages = [{"role": "system", "content": system}] + _work_api_messages_for(conv_id)
        try:
            text, tool_calls = agent.run_with_tools(full_messages, WORK_TOOLS, model)
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            return

        if tool_calls:
            call = tool_calls[0]
            name, args, call_id = call["name"], call["arguments"], call["id"]

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

            if name in WORK_READ_ONLY_TOOLS:
                try:
                    result = _execute_work_tool(name, args, workspace_root)
                except Exception as e:
                    result = f"Error: {e}"
                _add_tool_call_message(conv_id, call_id, name, args)
                _add_tool_result_message(conv_id, call_id, name, result)
                yield f"data: {json.dumps({'action': {'kind': name, 'args': args, 'result': result[:1200]}})}\n\n"
                continue

            if name in code_agent.APPROVAL_TOOLS:
                try:
                    description = code_agent.describe_pending(name, args, workspace_root)
                except Exception as e:
                    description = {"kind": name, "error": str(e)}
                _add_tool_call_message(conv_id, call_id, name, args)
                conv = models.Conversation.query.get(conv_id)
                _save_pending(conv, {"id": call_id, "name": name, "arguments": args})
                yield f"data: {json.dumps({'pending': description})}\n\n"
                return

            _add_tool_call_message(conv_id, call_id, name, args)
            _add_tool_result_message(conv_id, call_id, name, f"Error: unknown tool '{name}'.")
            continue

        assistant_msg = models.Message(conversation_id=conv_id, role="assistant", content=text or "")
        models.db.session.add(assistant_msg)
        models.db.session.commit()
        html_out = render_message(text or "")
        yield f"data: {json.dumps({'done': True, 'html': html_out})}\n\n"
        return

    yield f"data: {json.dumps({'error': 'Reached the max number of tool steps for this turn - try breaking the task down.'})}\n\n"


@app.route("/work", methods=["GET"])
def work_home():
    user = g.user
    conv_id = request.args.get("c", type=int)
    conv = models.Conversation.query.filter_by(id=conv_id, user_id=user.id, mode="work").first() if conv_id else None
    workspace_root = code_agent.user_workspace(user.id)

    if conv:
        transcript = _render_work_transcript(conv)
        pending = _pending(conv)
        pending_desc = code_agent.describe_pending(pending["name"], pending["arguments"], workspace_root) if pending else None
    else:
        conv_id = None
        transcript = []
        pending_desc = None

    sidebar = _sidebar_data(user, "work", conv_id)
    return render_template(
        "txlgpt_work.html",
        model_choices=MODEL_CHOICES,
        transcript=transcript,
        pending=pending_desc,
        active_id=conv_id,
        pinned=sidebar["pinned"],
        projects=sidebar["projects"],
        ungrouped=sidebar["ungrouped"],
        workspace=str(workspace_root),
        pygments_css=PYGMENTS_CSS,
        header_badge=HEADER_BADGE,
        user_email=user.email,
    )


@app.route("/work/send", methods=["POST"])
def work_send():
    user = g.user
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"error": "Please type a message."}), 400
    if len(message) > 8000:
        return jsonify({"error": "That message is too long (max 8000 characters)."}), 400

    model = pick_model(data, message)
    conv_id = data.get("conv_id")
    conv = models.Conversation.query.filter_by(id=conv_id, user_id=user.id, mode="work").first() if conv_id else None
    is_new = conv is None
    if is_new:
        conv = models.Conversation(user_id=user.id, mode="work", title=_make_title(message))
        models.db.session.add(conv)
        models.db.session.commit()
    elif _pending(conv):
        return jsonify({"error": "Resolve the pending action first."}), 409

    models.db.session.add(models.Message(conversation_id=conv.id, role="user", content=message))
    models.db.session.commit()
    conv_id, conv_title = conv.id, conv.title
    workspace_root = code_agent.user_workspace(user.id)

    def generate():
        if is_new:
            yield f"data: {json.dumps({'conv_id': conv_id, 'title': conv_title})}\n\n"
        yield from _run_work_loop(conv_id, model, workspace_root)

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


@app.route("/work/approve", methods=["POST"])
def work_approve():
    user = g.user
    data = request.get_json(silent=True) or {}
    conv = models.Conversation.query.filter_by(id=data.get("conv_id"), user_id=user.id, mode="work").first()
    if not conv:
        return jsonify({"error": "Not found"}), 404
    pending = _pending(conv)
    if not pending:
        return jsonify({"error": "Nothing pending."}), 400
    workspace_root = code_agent.user_workspace(user.id)
    try:
        result = code_agent.execute_pending(pending["name"], pending["arguments"], workspace_root)
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        result = f"Error: {e}"
    _add_tool_result_message(conv.id, pending["id"], pending["name"], result)
    _save_pending(conv, None)
    conv_id = conv.id
    model = ACCURATE_MODEL if data.get("model") == "auto" else pick_model(data)

    def generate():
        yield f"data: {json.dumps({'executed': result[:1200]})}\n\n"
        yield from _run_work_loop(conv_id, model, workspace_root)

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


@app.route("/work/deny", methods=["POST"])
def work_deny():
    user = g.user
    data = request.get_json(silent=True) or {}
    conv = models.Conversation.query.filter_by(id=data.get("conv_id"), user_id=user.id, mode="work").first()
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
    workspace_root = code_agent.user_workspace(user.id)
    return Response(
        stream_with_context(_run_work_loop(conv_id, model, workspace_root)),
        mimetype="text/event-stream",
    )


if __name__ == "__main__":
    port = int(os.environ.get("TXLGPT_PORT", 5003))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    # Local dev only binds to 127.0.0.1 (this machine only) - a real
    # deployment (Render) runs this via gunicorn instead, which binds
    # 0.0.0.0:$PORT itself (see render.yaml), never hitting this branch.
    app.run(host="127.0.0.1", port=port, debug=debug)
