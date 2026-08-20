"""
TXL Cloud - Password reset email
------------------------------------
Sends the "reset your password" email via plain SMTP - no third-party
email service/SDK needed, works with any SMTP account (e.g. a free
Gmail account with an "app password" - see README).

Configured entirely through environment variables:
  SMTP_HOST      e.g. smtp.gmail.com
  SMTP_PORT      default 587 (STARTTLS)
  SMTP_USER      the account to send from
  SMTP_PASSWORD  its password (an "app password" for Gmail, not your login one)
  SMTP_FROM      optional - defaults to SMTP_USER

If SMTP_HOST/SMTP_USER/SMTP_PASSWORD aren't all set, email sending is
considered unconfigured - see is_configured(). The /forgot-password route
checks this and tells the user plainly instead of pretending to send.
"""

import os
import smtplib
from email.mime.text import MIMEText


def is_configured() -> bool:
    return bool(os.environ.get("SMTP_HOST") and os.environ.get("SMTP_USER") and os.environ.get("SMTP_PASSWORD"))


def send_password_reset_email(to_email: str, reset_url: str) -> None:
    """Raises on failure - caller decides how to surface that."""
    host = os.environ["SMTP_HOST"]
    port = int(os.environ.get("SMTP_PORT", 587))
    user = os.environ["SMTP_USER"]
    password = os.environ["SMTP_PASSWORD"]
    from_addr = os.environ.get("SMTP_FROM", user)

    body = (
        "You asked to reset your TXL Cloud password.\n\n"
        f"Reset it here (link expires in 1 hour):\n{reset_url}\n\n"
        "If you didn't request this, you can safely ignore this email - "
        "your password hasn't been changed."
    )
    msg = MIMEText(body)
    msg["Subject"] = "Reset your TXL Cloud password"
    msg["From"] = from_addr
    msg["To"] = to_email

    with smtplib.SMTP(host, port, timeout=15) as server:
        server.starttls()
        server.login(user, password)
        server.sendmail(from_addr, [to_email], msg.as_string())
