"""
TXL Cloud - Database models
------------------------------------
Backs real user accounts and persisted chat/project data, so history
survives logins and server restarts - unlike the earlier in-memory,
no-account design.

Uses DATABASE_URL if set (e.g. a Neon/Postgres connection string in
production) - otherwise falls back to a local SQLite file
(txlclaude.db, next to this file) for development, no setup needed.
"""

import os
from datetime import datetime, timezone

from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import check_password_hash, generate_password_hash

db = SQLAlchemy()


def init_db(app):
    database_url = os.environ.get("DATABASE_URL", "sqlite:///txlclaude.db")
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
    """
    Lightweight, additive-only migration: adds columns introduced after
    the first release, so an existing database (and existing users' data)
    keeps working without a full migration framework. Safe to run on
    every startup - each statement is independently wrapped, so columns
    that already exist (including on a brand-new DB, where create_all()
    just made them) are silently skipped.
    """
    statements = [
        "ALTER TABLE conversations ADD COLUMN mode VARCHAR(10) DEFAULT 'chat'",
        "ALTER TABLE conversations ADD COLUMN pending_json TEXT",
        "ALTER TABLE messages ADD COLUMN tool_calls_json TEXT",
        "ALTER TABLE messages ADD COLUMN tool_call_id VARCHAR(64)",
        "ALTER TABLE messages ADD COLUMN tool_name VARCHAR(64)",
        "ALTER TABLE users ADD COLUMN reset_token VARCHAR(64)",
        "ALTER TABLE users ADD COLUMN reset_token_expires DATETIME",
    ]
    for stmt in statements:
        try:
            with engine.begin() as conn:
                conn.exec_driver_sql(stmt)
        except Exception:
            pass  # column already exists


def _utcnow():
    return datetime.now(timezone.utc)


class User(db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=_utcnow)
    # Forgot-password flow (see chat_app.py's /forgot-password, /reset-password):
    reset_token = db.Column(db.String(64), nullable=True)
    reset_token_expires = db.Column(db.DateTime, nullable=True)

    def set_password(self, password: str):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)


class Project(db.Model):
    __tablename__ = "projects"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    name = db.Column(db.String(120), nullable=False)
    created_at = db.Column(db.DateTime, default=_utcnow)
    files = db.relationship("ProjectFile", backref="project", cascade="all, delete-orphan", order_by="ProjectFile.id")


class ProjectFile(db.Model):
    """A reference file attached to a Project - its content is folded into
    the context of every chat inside that project (see chat_app.py's
    _project_knowledge_block()). Not real RAG/embeddings - just prepended
    text, which is fine at the small scale (a few KB) this is meant for."""
    __tablename__ = "project_files"
    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey("projects.id"), nullable=False, index=True)
    filename = db.Column(db.String(255), nullable=False)
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=_utcnow)


class Conversation(db.Model):
    __tablename__ = "conversations"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    project_id = db.Column(db.Integer, db.ForeignKey("projects.id"), nullable=True, index=True)
    mode = db.Column(db.String(10), nullable=False, default="chat", index=True)  # "chat" or "code"
    title = db.Column(db.String(200), nullable=False, default="New chat")
    pinned = db.Column(db.Boolean, default=False, nullable=False)
    pending_json = db.Column(db.Text, nullable=True)  # Code mode only: an approval awaiting the user
    created_at = db.Column(db.DateTime, default=_utcnow)
    messages = db.relationship(
        "Message", backref="conversation", cascade="all, delete-orphan",
        order_by="Message.id", lazy="joined",
    )


class Message(db.Model):
    __tablename__ = "messages"
    id = db.Column(db.Integer, primary_key=True)
    conversation_id = db.Column(db.Integer, db.ForeignKey("conversations.id"), nullable=False, index=True)
    role = db.Column(db.String(20), nullable=False)  # "user" / "assistant" / "tool"
    content = db.Column(db.Text, nullable=False, default="")
    # Code mode tool-calling (see chat_app.py's Code-mode routes):
    tool_calls_json = db.Column(db.Text, nullable=True)  # set on an assistant message proposing tool call(s)
    tool_call_id = db.Column(db.String(64), nullable=True)  # set on a "tool" role message (the result)
    tool_name = db.Column(db.String(64), nullable=True)     # set on a "tool" role message (the result)
    created_at = db.Column(db.DateTime, default=_utcnow)
