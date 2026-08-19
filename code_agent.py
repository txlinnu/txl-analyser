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
"""

import difflib
import os
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = Path(os.environ.get("CODE_WORKSPACE", BASE_DIR / "workspace")).resolve()
WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)

MAX_TOOL_STEPS = 8       # per user turn - guards against runaway tool-call loops
COMMAND_TIMEOUT = 60     # seconds
MAX_OUTPUT_CHARS = 4000
MAX_READ_CHARS = 20000

READ_ONLY_TOOLS = {"list_directory", "read_file"}
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
assumption - if you didn't call the tool, it didn't happen."""


def _safe_path(rel_path: str) -> Path:
    rel_path = (rel_path or ".").strip()
    target = (WORKSPACE_ROOT / rel_path).resolve()
    if target != WORKSPACE_ROOT and WORKSPACE_ROOT not in target.parents:
        raise ValueError(f"'{rel_path}' is outside the workspace and isn't allowed.")
    return target


def _tool_list_directory(args: dict) -> str:
    rel = args.get("path", ".")
    path = _safe_path(rel)
    if not path.exists():
        return f"Error: '{rel}' does not exist."
    if not path.is_dir():
        return f"Error: '{rel}' is not a directory."
    entries = sorted(os.listdir(path))
    if not entries:
        return "(empty directory)"
    return "\n".join((name + "/") if (path / name).is_dir() else name for name in entries)


def _tool_read_file(args: dict) -> str:
    rel = args.get("path", "")
    path = _safe_path(rel)
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


def execute_tool(name: str, args: dict) -> str:
    """Runs a read-only tool immediately. Never call this for write_file/run_command."""
    if name == "list_directory":
        return _tool_list_directory(args)
    if name == "read_file":
        return _tool_read_file(args)
    return f"Error: unknown tool '{name}'."


def describe_pending(name: str, args: dict) -> dict:
    """Human-readable description of a proposed write/exec action, for the approval UI."""
    if name == "write_file":
        rel_path = args.get("path", "")
        try:
            path = _safe_path(rel_path)
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


def execute_pending(name: str, args: dict) -> str:
    """Actually performs a write_file or run_command action. Only call after user approval."""
    if name == "write_file":
        path = _safe_path(args.get("path", ""))
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
                command, shell=True, cwd=WORKSPACE_ROOT,
                capture_output=True, text=True, timeout=COMMAND_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            return f"Error: command timed out after {COMMAND_TIMEOUT}s."
        except Exception as e:
            return f"Error running command: {e}"
        output = ((result.stdout or "") + (result.stderr or ""))[:MAX_OUTPUT_CHARS]
        return f"(exit code {result.returncode})\n{output}".strip()
    return f"Error: unknown tool '{name}'."
