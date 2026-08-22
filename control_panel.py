"""TXL Control Panel - a small webpage to manage the three TXL apps.

For each service you pick:
  - Online -> the real deployed website on Render. On/Off here actually
    suspends/resumes the Render service via Render's API (needs
    RENDER_API_KEY in .env). Works from anywhere this page is reachable.
  - Local -> runs the app right here on this PC. Has its own On/Off switch,
    and for TXL Cloud / TXL GPT a Groq/Ollama choice (Ollama is auto-started
    if it isn't already running). Only available when you open this page
    from the PC itself (127.0.0.1/localhost) - Local has no meaning from
    your phone, so those controls are hidden and blocked server-side too.

Run it and open the page it prints - everything else happens from there.

    python control_panel.py
"""

import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import requests
from flask import Flask, jsonify, render_template, request

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

app = Flask(__name__)

OLLAMA_HOST = "127.0.0.1"
OLLAMA_PORT = 11434

RENDER_API_KEY = os.environ.get("RENDER_API_KEY")
RENDER_API = "https://api.render.com/v1"

CREATIONFLAGS = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

# --- Service definitions ----------------------------------------------------

SERVICES = {
    "analyser": {
        "label": "TXL Analyser",
        "description": "Summarizes PDFs and YouTube videos.",
        "live_url": "https://txl-analyser.onrender.com",
        "render_service_id": "srv-da2nl6nlk1mc73cisj60",
        "backends": {
            "groq": {"script": "app.py", "port": 5000, "env": {}, "needs_ollama": False},
        },
        "default_backend": "groq",
    },
    "cloud": {
        "label": "TXL Cloud",
        "description": "Claude-style multi-turn chat assistant.",
        "live_url": "https://txl-cloud-z4p5.onrender.com",
        "render_service_id": "srv-da30q2ek1f9s73aq9g5g",
        "backends": {
            "groq": {"script": "chat_app.py", "port": 5001,
                     "env": {"CHAT_BACKEND": "groq", "CHAT_PORT": "5001"}, "needs_ollama": False},
            "ollama": {"script": "chat_app.py", "port": 5002,
                       "env": {"CHAT_BACKEND": "ollama", "CHAT_PORT": "5002"}, "needs_ollama": True},
        },
        "default_backend": "groq",
    },
    "gpt": {
        "label": "TXL GPT",
        "description": "ChatGPT-style app with accounts and history.",
        "live_url": "https://txl-gpt.onrender.com",
        "render_service_id": "srv-da4nsujbc2fs73bq0jhg",
        "backends": {
            "groq": {"script": "txlgpt_app.py", "port": 5003,
                     "env": {"TXLGPT_BACKEND": "groq", "TXLGPT_PORT": "5003"}, "needs_ollama": False},
            "ollama": {"script": "txlgpt_app.py", "port": 5004,
                       "env": {"TXLGPT_BACKEND": "ollama", "TXLGPT_PORT": "5004"}, "needs_ollama": True},
        },
        "default_backend": "groq",
    },
}

# --- In-memory state ---------------------------------------------------------
# local[key]  = {"proc", "backend", "status"}   status: stopped/starting/running/error
# online[key] = {"status"}  status: checking/live/waking/unreachable/suspended/not_deployed/error

_lock = threading.Lock()
local_state = {
    key: {"proc": None, "backend": cfg["default_backend"], "status": "stopped"}
    for key, cfg in SERVICES.items()
}
online_state = {
    key: {"status": "not_deployed" if not cfg["live_url"] else "checking"}
    for key, cfg in SERVICES.items()
}
location_state = {key: "local" for key in SERVICES}


def _port_open(port, host="127.0.0.1"):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex((host, port)) == 0


def is_local_request():
    return request.remote_addr in ("127.0.0.1", "::1")


# --- Ollama auto-start --------------------------------------------------------

def ensure_ollama_running(timeout=12):
    if _port_open(OLLAMA_PORT, OLLAMA_HOST):
        return True
    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=CREATIONFLAGS,
        )
    except FileNotFoundError:
        return False
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _port_open(OLLAMA_PORT, OLLAMA_HOST):
            return True
        time.sleep(0.5)
    return False


# --- Local process management -------------------------------------------------

def _monitor(key, proc, port, timeout=45):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            with _lock:
                if local_state[key]["proc"] is proc:
                    local_state[key]["status"] = "error"
            return
        if _port_open(port):
            with _lock:
                if local_state[key]["proc"] is proc:
                    local_state[key]["status"] = "running"
            return
        time.sleep(0.4)
    with _lock:
        if local_state[key]["proc"] is proc:
            local_state[key]["status"] = "error"


def start_local(key, backend):
    cfg = SERVICES[key]
    if backend not in cfg["backends"]:
        raise ValueError(f"unknown backend {backend!r} for {key!r}")

    stop_local(key)

    b = cfg["backends"][backend]
    log_path = LOG_DIR / f"{key}.log"

    if b["needs_ollama"]:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("Checking Ollama...\n")
        ok = ensure_ollama_running()
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("Ollama is up.\n" if ok else "Could not start Ollama automatically - is it installed?\n")

    env = os.environ.copy()
    env.pop("PORT", None)
    env.update(b["env"])

    logf = open(log_path, "a", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(
        [sys.executable, b["script"]],
        cwd=str(BASE_DIR),
        env=env,
        stdout=logf,
        stderr=subprocess.STDOUT,
        creationflags=CREATIONFLAGS,
    )

    with _lock:
        local_state[key] = {"proc": proc, "backend": backend, "status": "starting"}

    threading.Thread(target=_monitor, args=(key, proc, b["port"]), daemon=True).start()


def stop_local(key):
    with _lock:
        proc = local_state[key]["proc"]
        backend = local_state[key]["backend"]
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    with _lock:
        local_state[key] = {"proc": None, "backend": backend, "status": "stopped"}


def log_tail(key, n=25):
    path = LOG_DIR / f"{key}.log"
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-n:])


# --- Render API (online suspend/resume) --------------------------------------

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
        data = r.json()
        if data.get("suspended") == "suspended":
            with _lock:
                online_state[key]["status"] = "suspended"
            return
    except requests.RequestException:
        with _lock:
            online_state[key]["status"] = "error"
        return

    # Not suspended - check whether it's actually answering (it may still be asleep/free-tier)
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


def _online_poll_loop():
    while True:
        for key, cfg in SERVICES.items():
            if cfg["live_url"]:
                _check_online(key, cfg)
        with _lock:
            fast = any(online_state[k]["status"] in ("waking", "unreachable", "checking", "error") for k in online_state)
        time.sleep(10 if fast else 30)


threading.Thread(target=_online_poll_loop, daemon=True).start()


# --- Snapshot for the frontend -------------------------------------------------

def snapshot():
    with _lock:
        out = {}
        for key, cfg in SERVICES.items():
            ls = local_state[key]
            out[key] = {
                "label": cfg["label"],
                "description": cfg["description"],
                "backends": list(cfg["backends"].keys()),
                "live_url": cfg["live_url"],
                "location": location_state[key],
                "local": {
                    "backend": ls["backend"],
                    "status": ls["status"],
                    "port": cfg["backends"][ls["backend"]]["port"],
                },
                "online": dict(online_state[key]),
            }
        return out


# --- Routes --------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("control_panel.html")


@app.route("/api/role")
def api_role():
    return jsonify({"role": "full" if is_local_request() else "online_only"})


@app.route("/api/status")
def api_status():
    return jsonify(snapshot())


@app.route("/api/location", methods=["POST"])
def api_location():
    data = request.get_json(force=True)
    key = data.get("service")
    location = data.get("location")
    if key not in SERVICES or location not in ("local", "online"):
        return jsonify({"error": "bad request"}), 400
    if location == "local" and not is_local_request():
        return jsonify({"error": "Local control is only available from this PC"}), 403
    location_state[key] = location
    return jsonify(snapshot())


@app.route("/api/start", methods=["POST"])
def api_start():
    if not is_local_request():
        return jsonify({"error": "Local control is only available from this PC"}), 403
    data = request.get_json(force=True)
    key = data.get("service")
    backend = data.get("backend")
    if key not in SERVICES:
        return jsonify({"error": "unknown service"}), 400
    backend = backend or local_state[key]["backend"]
    start_local(key, backend)
    return jsonify(snapshot())


@app.route("/api/stop", methods=["POST"])
def api_stop():
    if not is_local_request():
        return jsonify({"error": "Local control is only available from this PC"}), 403
    data = request.get_json(force=True)
    key = data.get("service")
    if key not in SERVICES:
        return jsonify({"error": "unknown service"}), 400
    stop_local(key)
    return jsonify(snapshot())


@app.route("/api/render/<action>", methods=["POST"])
def api_render_action(action):
    if action not in ("suspend", "resume"):
        return jsonify({"error": "unknown action"}), 400
    data = request.get_json(force=True)
    key = data.get("service")
    if key not in SERVICES or not SERVICES[key]["live_url"]:
        return jsonify({"error": "unknown or undeployed service"}), 400
    if not RENDER_API_KEY:
        return jsonify({"error": "RENDER_API_KEY not configured"}), 500

    service_id = SERVICES[key]["render_service_id"]
    ok = render_suspend(service_id) if action == "suspend" else render_resume(service_id)
    with _lock:
        online_state[key]["status"] = "suspended" if (action == "suspend" and ok) else "checking"
    return jsonify({"ok": ok, **snapshot()})


@app.route("/api/log/<key>")
def api_log(key):
    if key not in SERVICES:
        return jsonify({"error": "unknown service"}), 400
    return jsonify({"log": log_tail(key)})


if __name__ == "__main__":
    port = int(os.environ.get("CONTROL_PORT", 5099))
    host = os.environ.get("CONTROL_HOST", "0.0.0.0")
    print(f" * TXL Control Panel: http://127.0.0.1:{port}  (also listening on {host})")
    app.run(host=host, port=port, debug=False)
