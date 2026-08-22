"""
Txl GPT - Database models
------------------------------------
Standalone SQLite-backed models for Txl GPT - a separate app from TXL
Cloud (chat_app.py/models.py): its own tables, own database file
(txlgpt.db), own accounts, so the two run side by side without sharing
any data.
"""

import os
import sys
from datetime import datetime, timezone

from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import check_password_hash, generate_password_hash

db = SQLAlchemy()


def init_db(app):
    database_url = os.environ.get("TXLGPT_DATABASE_URL", "sqlite:///txlgpt.db")
    # Some hosts (Render, Heroku-style) hand out postgres:// - SQLAlchemy 1.4+/2.x wants postgresql://
    if database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql://", 1)
    app.config["SQLALCHEMY_DATABASE_URI"] = database_url
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    db.init_app(app)
    with app.app_context():
        db.create_all()
        _migrate(db.engine)


def _migrate(engine):
    """Lightweight, additive-only migration: adds columns introduced after
    the first release, so an existing database keeps working without a
    full migration framework. Safe to run on every startup - each
    statement is independently wrapped, so columns that already exist
    (including on a brand-new DB, where create_all() just made them) are
    silently skipped."""
    statements = [
        "ALTER TABLE txlgpt_users ADD COLUMN reset_token VARCHAR(64)",
        "ALTER TABLE txlgpt_users ADD COLUMN reset_token_expires TIMESTAMP",
        "ALTER TABLE txlgpt_conversations ADD COLUMN pinned BOOLEAN DEFAULT 0",
        "ALTER TABLE txlgpt_conversations ADD COLUMN project_id INTEGER",
        "ALTER TABLE txlgpt_conversations ADD COLUMN mode VARCHAR(10) DEFAULT 'chat'",
        "ALTER TABLE txlgpt_conversations ADD COLUMN gpt_id INTEGER",
        "ALTER TABLE txlgpt_conversations ADD COLUMN pending_json TEXT",
        "ALTER TABLE txlgpt_messages ADD COLUMN tool_calls_json TEXT",
        "ALTER TABLE txlgpt_messages ADD COLUMN tool_call_id VARCHAR(64)",
        "ALTER TABLE txlgpt_messages ADD COLUMN tool_name VARCHAR(64)",
    ]
    for stmt in statements:
        try:
            with engine.begin() as conn:
                conn.exec_driver_sql(stmt)
        except Exception as e:
            if "already exists" not in str(e).lower() and "duplicate column" not in str(e).lower():
                print(f"[migrate] warning: {stmt!r} failed: {e}", file=sys.stderr)


def _utcnow():
    return datetime.now(timezone.utc)


class User(db.Model):
    __tablename__ = "txlgpt_users"
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=_utcnow)
    # Forgot-password flow (see txlgpt_app.py's /forgot-password, /reset-password):
    reset_token = db.Column(db.String(64), nullable=True)
    reset_token_expires = db.Column(db.DateTime, nullable=True)

    def set_password(self, password: str):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)


class Project(db.Model):
    __tablename__ = "txlgpt_projects"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("txlgpt_users.id"), nullable=False, index=True)
    name = db.Column(db.String(120), nullable=False)
    created_at = db.Column(db.DateTime, default=_utcnow)


class CustomGPT(db.Model):
    """A user-defined persona: a name + standing instructions that replace
    the default system prompt for any chat started under it."""
    __tablename__ = "txlgpt_custom_gpts"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("txlgpt_users.id"), nullable=False, index=True)
    name = db.Column(db.String(80), nullable=False)
    description = db.Column(db.String(200), nullable=True)
    instructions = db.Column(db.Text, nullable=False)
    icon = db.Column(db.String(8), nullable=False, default="🤖")
    created_at = db.Column(db.DateTime, default=_utcnow)


class Memory(db.Model):
    """A fact Txl GPT should remember about this user across every future
    conversation - either added by the user directly, or saved by the
    model mid-chat via the 'remember' tool (see txlgpt_app.py)."""
    __tablename__ = "txlgpt_memories"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("txlgpt_users.id"), nullable=False, index=True)
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=_utcnow)


class Conversation(db.Model):
    __tablename__ = "txlgpt_conversations"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("txlgpt_users.id"), nullable=False, index=True)
    project_id = db.Column(db.Integer, db.ForeignKey("txlgpt_projects.id"), nullable=True, index=True)
    gpt_id = db.Column(db.Integer, db.ForeignKey("txlgpt_custom_gpts.id"), nullable=True, index=True)
    mode = db.Column(db.String(10), nullable=False, default="chat")  # "chat" or "code"
    title = db.Column(db.String(200), nullable=False, default="New chat")
    pinned = db.Column(db.Boolean, default=False, nullable=False)
    pending_json = db.Column(db.Text, nullable=True)  # Code/Work mode only: a tool call awaiting approval
    created_at = db.Column(db.DateTime, default=_utcnow)
    messages = db.relationship(
        "Message", backref="conversation", cascade="all, delete-orphan",
        order_by="Message.id", lazy="joined",
    )


class Message(db.Model):
    __tablename__ = "txlgpt_messages"
    id = db.Column(db.Integer, primary_key=True)
    conversation_id = db.Column(db.Integer, db.ForeignKey("txlgpt_conversations.id"), nullable=False, index=True)
    role = db.Column(db.String(20), nullable=False)  # "user" / "assistant" / "tool"
    content = db.Column(db.Text, nullable=False, default="")
    image_data = db.Column(db.Text, nullable=True)  # an attached/generated image, as a data: URL
    # Work/Code mode tool-calling (see txlgpt_app.py's /work/* routes):
    tool_calls_json = db.Column(db.Text, nullable=True)  # set on an assistant message proposing tool call(s)
    tool_call_id = db.Column(db.String(64), nullable=True)  # set on a "tool" role message (the result)
    tool_name = db.Column(db.String(64), nullable=True)     # set on a "tool" role message (the result)
    created_at = db.Column(db.DateTime, default=_utcnow)


class ScheduledTask(db.Model):
    """A prompt the user wants run automatically, on its own schedule -
    see the background poller in txlgpt_app.py (_scheduler_loop) that
    executes due tasks and drops the result into a conversation."""
    __tablename__ = "txlgpt_scheduled_tasks"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("txlgpt_users.id"), nullable=False, index=True)
    prompt = db.Column(db.Text, nullable=False)
    schedule_type = db.Column(db.String(10), nullable=False)  # "once" / "daily" / "interval"
    run_at_time = db.Column(db.String(5), nullable=True)      # "daily": "HH:MM" (UTC)
    interval_minutes = db.Column(db.Integer, nullable=True)   # "interval": minutes between runs
    next_run = db.Column(db.DateTime, nullable=False, index=True)
    last_run = db.Column(db.DateTime, nullable=True)
    conversation_id = db.Column(db.Integer, db.ForeignKey("txlgpt_conversations.id"), nullable=True)
    enabled = db.Column(db.Boolean, default=True, nullable=False)
    created_at = db.Column(db.DateTime, default=_utcnow)
