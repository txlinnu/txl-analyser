"""TXL Remote Control - a tiny, always-on webpage for suspending/resuming
the three TXL Render services from anywhere, regardless of whether your PC
is on. Deployed on Render itself (see render.yaml), separate from
control_panel.py (which also runs the Local, on-this-PC side of things and
therefore can only ever work while the PC is on).

Password-gated with REMOTE_CONTROL_PASSWORD (same pattern as this repo's
existing SITE_PASSWORD gate) since this one is a real public URL.
"""

import os
import threading
import time

import requests
from flask import Flask, Response, jsonify, render_template, request

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = Flask(__name__)

REMOTE_CONTROL_PASSWORD = os.environ.get("REMOTE_CONTROL_PASSWORD")
RENDER_API_KEY = os.environ.get("RENDER_API_KEY")
RENDER_API = "https://api.render.com/v1"

SERVICES = {
    "analyser": {
        "label": "TXL Analyser",
        "description": "Summarizes PDFs and YouTube videos.",
        "live_url": "https://txl-analyser.onrender.com",
        "render_service_id": "srv-da2nl6nlk1mc73cisj60",
    },
    "cloud": {
        "label": "TXL Cloud",
        "description": "Claude-style multi-turn chat assistant.",
        "live_url": "https://txl-cloud-z4p5.onrender.com",
        "render_service_id": "srv-da30q2ek1f9s73aq9g5g",
    },
    "gpt": {
        "label": "TXL GPT",
        "description": "ChatGPT-style app with accounts and history.",
        "live_url": "https://txl-gpt.onrender.com",
        "render_service_id": "srv-da4nsujbc2fs73bq0jhg",
    },
}


@app.before_request
def require_password():
    if not REMOTE_CONTROL_PASSWORD:
        return None
    auth = request.authorization
    if not auth or auth.password != REMOTE_CONTROL_PASSWORD:
        return Response(
            "Login required", 401, {"WWW-Authenticate": 'Basic realm="TXL Remote Control"'}
        )
    return None


# --- Render API ---------------------------------------------------------------

_lock = threading.Lock()
online_state = {key: {"status": "checking"} for key in SERVICES}


def _render_headers():
    return {"Authorization": f"Bearer {RENDER_API_KEY}", "Accept": "application/json"}


def render_suspend(service_id):
    r = requests.post(f"{RENDER_API}/services/{service_id}/suspend", headers=_render_headers(), timeout=15)
    return r.status_code < 300


def render_resume(service_id):
    r = requests.post(f"{RENDER_API}/services/{service_id}/resume", headers=_render_headers(), timeout=15)
    return r.status_code < 300


def _check_online(key, cfg):
    if not RENDER_API_KEY:
        with _lock:
            online_state[key]["status"] = "error"
        return
    try:
        r = requests.get(f"{RENDER_API}/services/{cfg['render_service_id']}", headers=_render_headers(), timeout=10)
        if r.status_code >= 300:
            with _lock:
                online_state[key]["status"] = "error"
            return
        if r.json().get("suspended") == "suspended":
            with _lock:
                online_state[key]["status"] = "suspended"
            return
    except requests.RequestException:
        with _lock:
            online_state[key]["status"] = "error"
        return

    try:
        resp = requests.head(cfg["live_url"], timeout=25, allow_redirects=True)
        if resp.status_code == 503:
            status = "waking"
        elif resp.status_code < 500:
            status = "live"
        else:
            status = "unreachable"
    except requests.RequestException:
        status = "unreachable"
    with _lock:
        online_state[key]["status"] = status


def _poll_loop():
    while True:
        for key, cfg in SERVICES.items():
            _check_online(key, cfg)
        with _lock:
            fast = any(online_state[k]["status"] in ("waking", "unreachable", "checking", "error") for k in online_state)
        time.sleep(10 if fast else 30)


_poll_started = False
_poll_start_lock = threading.Lock()


def _ensure_poll_started():
    # Started lazily on first request (not at import time) so it reliably
    # runs inside the actual worker process under gunicorn, regardless of
    # whether the app was imported pre- or post-fork.
    global _poll_started
    if _poll_started:
        return
    with _poll_start_lock:
        if not _poll_started:
            threading.Thread(target=_poll_loop, daemon=True).start()
            _poll_started = True


@app.before_request
def _start_poll():
    _ensure_poll_started()


def snapshot():
    with _lock:
        return {
            key: {
                "label": cfg["label"],
                "description": cfg["description"],
                "live_url": cfg["live_url"],
                "online": dict(online_state[key]),
            }
            for key, cfg in SERVICES.items()
        }


# --- Routes --------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("render_control.html")


@app.route("/api/status")
def api_status():
    return jsonify(snapshot())


@app.route("/api/render/<action>", methods=["POST"])
def api_render_action(action):
    if action not in ("suspend", "resume"):
        return jsonify({"error": "unknown action"}), 400
    data = request.get_json(force=True)
    key = data.get("service")
    if key not in SERVICES:
        return jsonify({"error": "unknown service"}), 400
    if not RENDER_API_KEY:
        return jsonify({"error": "RENDER_API_KEY not configured"}), 500

    service_id = SERVICES[key]["render_service_id"]
    ok = render_suspend(service_id) if action == "suspend" else render_resume(service_id)
    with _lock:
        online_state[key]["status"] = "suspended" if (action == "suspend" and ok) else "checking"
    return jsonify({"ok": ok, **snapshot()})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5098))
    app.run(host="0.0.0.0", port=port, debug=False)
