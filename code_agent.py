"""
TXL Cloud - Code Mode engine
------------------------------------
Turns TXL Cloud into a small Claude-Code-style coding agent: it can list
directories, read files, write files, and run shell commands inside a
sandboxed workspace folder - using whichever chat backend (Groq or
Ollama) chat_app.py currently has loaded as `agent`.

Safety model:
  - Read-only tools (list_directory, read_file) run automatically.
  - Anything that touches disk or runs a command (write_file, run_command)
    is only *proposed* - it does NOT execute until the user clicks Approve
    in the UI. See describe_pending() / execute_pending().
  - Nothing outside WORKSPACE_ROOT is reachable. Paths that try to escape
    it (../, absolute paths elsewhere) are rejected.

Workspace: defaults to a workspace/ folder next to this file, so the
agent can't accidentally touch this app's own source. Point it at a real
project instead by setting CODE_WORKSPACE to that folder's path.

Multi-account isolation: this app has real user accounts (models.py), so
every tool call is scoped to a per-user subfolder via user_workspace(uid)
- WORKSPACE_ROOT/user_<id>/ - never the shared root directly. One
account's files/commands can never see or touch another's.
"""

import difflib
import os
import re
import string
import subprocess
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = Path(os.environ.get("CODE_WORKSPACE", BASE_DIR / "workspace")).resolve()
WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)

# Off by default (and MUST stay off on the shared live site) - lets Code
# mode point at an arbitrary real folder on disk instead of the sandboxed
# per-user workspace. Only safe to enable on a single-user local install,
# where "arbitrary folder on disk" just means the owner's own machine.
# Set ALLOW_CUSTOM_WORKSPACE=1 in that local deployment's environment only.
ALLOW_CUSTOM_WORKSPACE = os.environ.get("ALLOW_CUSTOM_WORKSPACE", "0") == "1"

_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def user_workspace(user_id) -> Path:
    """Each account gets its own isolated subfolder - never share WORKSPACE_ROOT directly."""
    uid = str(user_id)
    if not _SAFE_ID_RE.match(uid):
        raise ValueError("Invalid user id.")
    path = WORKSPACE_ROOT / f"user_{uid}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_workspace(user_id, custom_path: str = None) -> Path:
    """Same as user_workspace(), but honors a per-conversation custom folder
    path when ALLOW_CUSTOM_WORKSPACE is enabled - falls back to the normal
    sandboxed workspace otherwise (including when the flag is off, so a
    stray custom_path from an old conversation can never do anything on a
    deployment where this wasn't explicitly turned on)."""
    if ALLOW_CUSTOM_WORKSPACE and custom_path:
        path = Path(custom_path).resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path
    return user_workspace(user_id)


def validate_custom_workspace(raw_path: str) -> Path:
    """Validates a user-supplied folder path for use as a custom workspace.
    Raises ValueError with a user-facing message on anything invalid."""
    if not ALLOW_CUSTOM_WORKSPACE:
        raise ValueError("Custom project folders aren't enabled on this deployment.")
    raw_path = (raw_path or "").strip()
    if not raw_path:
        raise ValueError("Enter a folder path.")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        raise ValueError("Use a full path (e.g. C:\\Projects\\my-app or /home/you/my-app).")
    if path.exists() and not path.is_dir():
        raise ValueError(f"'{raw_path}' exists but isn't a folder.")
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def _list_drives() -> list:
    """Windows: available drive letters. Unix: just root."""
    if os.name == "nt":
        return [f"{letter}:\\" for letter in string.ascii_uppercase if Path(f"{letter}:\\").exists()]
    return ["/"]


def browse_folders(path: str = None) -> dict:
    """Lists subfolders of `path`, for the custom-workspace folder-picker UI.
    path=None/"" means the top level (drives on Windows, / on Unix). Returns
    {"path", "parent", "folders": [{"name", "path"}, ...]} - "parent" is ""
    for the top level, None only never (there's always somewhere to go up
    to, down to the top level itself)."""
    if not ALLOW_CUSTOM_WORKSPACE:
        raise ValueError("Custom project folders aren't enabled on this deployment.")
    if not path:
        drives = _list_drives()
        return {"path": "", "parent": None, "folders": [{"name": d, "path": d} for d in drives]}

    p = Path(path).expanduser()
    if not p.is_absolute():
        raise ValueError("Invalid path.")
    if not p.exists() or not p.is_dir():
        raise ValueError(f"'{path}' doesn't exist or isn't a folder.")
    p = p.resolve()

    _hidden = {"$recycle.bin", "system volume information"}
    try:
        entries = sorted(
            (e for e in p.iterdir() if e.is_dir() and not e.name.startswith(".") and e.name.lower() not in _hidden),
            key=lambda e: e.name.lower(),
        )
    except PermissionError:
        entries = []
    folders = [{"name": e.name, "path": str(e)} for e in entries]
    is_drive_root = os.name == "nt" and p.parent == p
    parent = "" if is_drive_root else str(p.parent)
    return {"path": str(p), "parent": parent, "folders": folders}


MAX_TOOL_STEPS = 8       # per user turn - guards against runaway tool-call loops
COMMAND_TIMEOUT = 60     # seconds
MAX_OUTPUT_CHARS = 4000
MAX_READ_CHARS = 20000
FETCH_TIMEOUT = 15       # seconds
MAX_FETCH_CHARS = 8000

READ_ONLY_TOOLS = {"list_directory", "read_file", "fetch_url"}
APPROVAL_TOOLS = {"write_file", "run_command"}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files and folders in a directory inside the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to the workspace root. Use '.' for the root."}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the full text content of a file inside the workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Path relative to the workspace root."}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": (
                "Fetch a URL over HTTP(S) and return its response - e.g. to check "
                "whether a locally running dev server is responding, or to read a "
                "webpage. Works for localhost/127.0.0.1 URLs (like the app you're "
                "currently building) as well as the public internet. Read-only, "
                "runs immediately without approval."
            ),
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string", "description": "The full URL, including http:// or https://."}},
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Create or overwrite a file inside the workspace with new content. "
                "Requires the user's explicit approval before it actually runs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to the workspace root."},
                    "content": {"type": "string", "description": "The full new content of the file."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": (
                "Run a shell command in the workspace directory. "
                "Requires the user's explicit approval before it actually runs."
            ),
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string", "description": "The exact shell command to run."}},
                "required": ["command"],
            },
        },
    },
]

SYSTEM_PROMPT_TEMPLATE = """You are TXL Cloud in Code mode: a coding \
agent with tools to inspect and modify files, and run shell commands, \
inside the user's local workspace at {workspace}. You cannot see or \
touch anything outside that folder.

- Use list_directory and read_file to look around before making changes \
- don't guess at file contents.
- Use fetch_url to check whether something you built (or anything else, \
including localhost/127.0.0.1 URLs) is actually responding, and what it \
returns - don't say you "can't access the browser" or guess at what a \
running server does, actually fetch it.
- Use write_file to create or edit files, and run_command for terminal \
commands (installing packages, running scripts/tests, git, etc.).
- write_file and run_command both require the user's explicit approval \
before they actually execute - after you call one, wait for the tool \
result to tell you whether it was approved or denied. Never assume it \
already happened.
- Be concise. Let the actions speak; don't narrate every step at length.
- If the user denies an action, don't retry the same thing - ask what \
they'd prefer instead.
- Never say a file was written or a command was run unless you actually \
called write_file/run_command for it and got back a tool result \
confirming it. Do not describe an action as done from memory or \
assumption - if you didn't call the tool, it didn't happen.

When building a website or UI, treat it as production/premium work, not \
a rough sketch - the user is building something they intend to actually \
use, not a placeholder:
- Write real CSS (a separate .css file for anything beyond a single page)
  with intentional typography (a real font stack, sensible size/weight
  scale), consistent spacing, and a considered color palette - never bare
  unstyled HTML tags with no visual design at all.
- Use modern layout (flexbox/grid), make it responsive (it should not
  break or look broken on a phone-width screen), and use semantic HTML5
  elements (<header>, <nav>, <main>, <section>, <footer>) instead of
  nesting everything in generic <div>s.
- Add real interaction/motion where it genuinely improves the result
  (hover states, smooth transitions, an active nav state) - subtle and
  purposeful, not decoration for its own sake.
- Multi-page sites should share one consistent header/nav/footer and
  visual language across every page, not look like separate unrelated
  pages.
- If the user's request is vague on style, default to a clean, modern,
  professional look rather than the plainest possible interpretation -
  ask what aesthetic they want only if it's genuinely unclear which
  direction to take, not for routine requests.
- If an HTML file references another file (a stylesheet, a script, an
  image), that file must actually exist - write it in the same turn.
  Never leave a <link>/<script src>/<img> pointing at something you
  didn't create; a page that references a missing file will load broken
  and unstyled. If you'd rather keep it to one file, inline the CSS in a
  <style> tag instead of linking a separate stylesheet you don't write."""


def _safe_path(rel_path: str, workspace_root: Path) -> Path:
    rel_path = (rel_path or ".").strip()
    target = (workspace_root / rel_path).resolve()
    if target != workspace_root and workspace_root not in target.parents:
        raise ValueError(f"'{rel_path}' is outside the workspace and isn't allowed.")
    return target


def _tool_list_directory(args: dict, workspace_root: Path) -> str:
    rel = args.get("path", ".")
    path = _safe_path(rel, workspace_root)
    if not path.exists():
        return f"Error: '{rel}' does not exist."
    if not path.is_dir():
        return f"Error: '{rel}' is not a directory."
    entries = sorted(os.listdir(path))
    if not entries:
        return "(empty directory)"
    return "\n".join((name + "/") if (path / name).is_dir() else name for name in entries)


def _tool_read_file(args: dict, workspace_root: Path) -> str:
    rel = args.get("path", "")
    path = _safe_path(rel, workspace_root)
    if not path.exists():
        return f"Error: '{rel}' does not exist."
    if not path.is_file():
        return f"Error: '{rel}' is not a file."
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"Error reading file: {e}"
    if len(text) > MAX_READ_CHARS:
        text = text[:MAX_READ_CHARS] + f"\n... (truncated, {len(text):,} characters total)"
    return text


def _tool_fetch_url(args: dict) -> str:
    url = (args.get("url") or "").strip()
    if not re.match(r"^https?://", url, re.IGNORECASE):
        return "Error: url must start with http:// or https://."
    try:
        resp = requests.get(url, timeout=FETCH_TIMEOUT, headers={"User-Agent": "TXL-Cloud-CodeMode/1.0"})
    except requests.exceptions.ConnectionError as e:
        return f"Error: could not connect to {url} - is anything actually listening there? ({e})"
    except requests.exceptions.Timeout:
        return f"Error: {url} took too long to respond (timed out after {FETCH_TIMEOUT}s)."
    except Exception as e:
        return f"Error fetching {url}: {e}"

    content_type = resp.headers.get("Content-Type", "unknown")
    body = resp.text
    if len(body) > MAX_FETCH_CHARS:
        body = body[:MAX_FETCH_CHARS] + f"\n... (truncated, {len(body):,} characters total)"
    return f"HTTP {resp.status_code} ({content_type})\n\n{body}"


def execute_tool(name: str, args: dict, workspace_root: Path) -> str:
    """Runs a read-only tool immediately. Never call this for write_file/run_command."""
    if name == "list_directory":
        return _tool_list_directory(args, workspace_root)
    if name == "read_file":
        return _tool_read_file(args, workspace_root)
    if name == "fetch_url":
        return _tool_fetch_url(args)
    return f"Error: unknown tool '{name}'."


def describe_pending(name: str, args: dict, workspace_root: Path) -> dict:
    """Human-readable description of a proposed write/exec action, for the approval UI."""
    if name == "write_file":
        rel_path = args.get("path", "")
        try:
            path = _safe_path(rel_path, workspace_root)
        except ValueError as e:
            return {"kind": "write_file", "path": rel_path, "error": str(e)}
        old = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
        new = args.get("content", "")
        diff_lines = list(difflib.unified_diff(
            old.splitlines(), new.splitlines(),
            fromfile=f"{rel_path} (current)", tofile=f"{rel_path} (proposed)", lineterm="",
        ))
        return {
            "kind": "write_file",
            "path": rel_path,
            "is_new": not path.exists(),
            "diff": "\n".join(diff_lines) if diff_lines else "(no textual change)",
        }
    if name == "run_command":
        return {"kind": "run_command", "command": args.get("command", "")}
    return {"kind": name, "args": args}


def execute_pending(name: str, args: dict, workspace_root: Path) -> str:
    """Actually performs a write_file or run_command action. Only call after user approval."""
    if name == "write_file":
        path = _safe_path(args.get("path", ""), workspace_root)
        content = args.get("content", "")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"Wrote {len(content):,} characters to {args.get('path')}."
    if name == "run_command":
        command = (args.get("command") or "").strip()
        if not command:
            return "Error: empty command."
        try:
            result = subprocess.run(
                command, shell=True, cwd=workspace_root,
                capture_output=True, text=True, timeout=COMMAND_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            return f"Error: command timed out after {COMMAND_TIMEOUT}s."
        except Exception as e:
            return f"Error running command: {e}"
        output = ((result.stdout or "") + (result.stderr or ""))[:MAX_OUTPUT_CHARS]
        return f"(exit code {result.returncode})\n{output}".strip()
    return f"Error: unknown tool '{name}'."
