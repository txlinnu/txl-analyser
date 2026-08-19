"""
TXL Analyser - Web UI
------------------------------------
A small local Flask website in front of the two free/cloud agents:
  - pdf_research_agent_local.py  (summarize + research a PDF's topics)
  - youtube_agent.py             (summarize a YouTube video from its transcript)

Uses Groq's free API for inference (no local model, doesn't load down your
PC) - no cost, but requires a free GROQ_API_KEY (see README.md).

Run:
    python app.py
Then open http://127.0.0.1:5000 in your browser.
"""

import os
import sys
import traceback
import uuid
from pathlib import Path

import markdown as md
from flask import Flask, Response, render_template, request
from werkzeug.utils import secure_filename

import pdf_research_agent_local as pdf_agent
import youtube_agent

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB upload cap

# Optional password gate - set SITE_PASSWORD in the environment (e.g. on Render)
# to require a login before anyone can use a public deployment. Unset = no gate
# (fine for local-only use, where 127.0.0.1 binding already keeps it private).
SITE_PASSWORD = os.environ.get("SITE_PASSWORD")


@app.before_request
def require_password():
    if not SITE_PASSWORD:
        return None
    auth = request.authorization
    if not auth or auth.password != SITE_PASSWORD:
        return Response(
            "Login required", 401, {"WWW-Authenticate": 'Basic realm="TXL Analyser"'}
        )
    return None

# (model name, label shown in the dropdown)
MODEL_CHOICES = [
    ("openai/gpt-oss-120b", "Balanced — more accurate, slower"),
    ("openai/gpt-oss-20b", "Fast — quicker, less detailed"),
]
VALID_MODELS = {m for m, _ in MODEL_CHOICES}


def pick_model(form) -> str:
    model = form.get("model", "")
    return model if model in VALID_MODELS else pdf_agent.DEFAULT_MODEL


def render_report(kind: str, title: str, report_markdown: str):
    html = md.markdown(report_markdown, extensions=["extra", "sane_lists"])
    return render_template(
        "index.html", model_choices=MODEL_CHOICES,
        result_kind=kind, result_title=title, result_html=html,
    )


def render_error(message: str):
    return render_template("index.html", model_choices=MODEL_CHOICES, error=message)


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", model_choices=MODEL_CHOICES)


@app.route("/pdf", methods=["POST"])
def process_pdf():
    file = request.files.get("pdf_file")
    if not file or file.filename == "":
        return render_error("Please choose a PDF file.")
    if not file.filename.lower().endswith(".pdf"):
        return render_error("That doesn't look like a PDF file.")

    model = pick_model(request.form)
    safe_name = secure_filename(file.filename) or "upload.pdf"
    temp_path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{safe_name}"
    file.save(temp_path)

    try:
        pdf_text, truncated = pdf_agent.extract_pdf_text(temp_path)
        report = pdf_agent.run_agent(pdf_text, model)
        if truncated:
            report = f"_Note: this PDF was long, so only the first ~{pdf_agent.MAX_PDF_CHARS:,} characters were used._\n\n" + report
        return render_report("pdf", file.filename, report)
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        return render_error(f"Something went wrong processing the PDF: {e}")
    finally:
        temp_path.unlink(missing_ok=True)


@app.route("/youtube", methods=["POST"])
def process_youtube():
    url = (request.form.get("youtube_url") or "").strip()
    if not url:
        return render_error("Please paste a YouTube URL.")

    transcript_text = (request.form.get("transcript_text") or "").strip() or None
    model = pick_model(request.form)
    try:
        report = youtube_agent.run(url, model, transcript_text=transcript_text)
        return render_report("youtube", url, report)
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        return render_error(f"Something went wrong summarizing that video: {e}")


if __name__ == "__main__":
    # Local runs stay on 127.0.0.1 (private, this machine only). A production
    # deploy (e.g. Render) runs this via gunicorn instead, which handles its
    # own host/port binding - see render.yaml.
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=False)
