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

When SMTP_HOST is smtp.sendgrid.net, sending goes over SendGrid's HTTPS
API instead of raw SMTP. Reason: PaaS hosts like Render commonly block or
silently blackhole outbound SMTP ports (587/25) to stop spam relaying, so
a plain smtplib connection just hangs instead of failing - discovered
when the live Render deployment hung for 10+ minutes on every reset
request while the exact same SMTP credentials worked fine locally. The
SendGrid API key (SMTP_PASSWORD, since SMTP_USER is literally "apikey"
for SendGrid) works unchanged for both the SMTP and API auth methods.
"""

import os
import smtplib
from email.mime.text import MIMEText

import requests


def is_configured() -> bool:
    return bool(os.environ.get("SMTP_HOST") and os.environ.get("SMTP_USER") and os.environ.get("SMTP_PASSWORD"))


def send_password_reset_email(to_email: str, reset_url: str, product_name: str = "TXL Cloud") -> None:
    """Raises on failure - caller decides how to surface that. `product_name`
    lets other apps (e.g. txlgpt_app.py) reuse this same SMTP sender with
    their own branding in the email text."""
    host = os.environ["SMTP_HOST"]
    user = os.environ["SMTP_USER"]
    password = os.environ["SMTP_PASSWORD"]
    from_addr = os.environ.get("SMTP_FROM", user)

    subject = f"Reset your {product_name} password"
    body = (
        f"You asked to reset your {product_name} password.\n\n"
        f"Reset it here (link expires in 1 hour):\n{reset_url}\n\n"
        "If you didn't request this, you can safely ignore this email - "
        "your password hasn't been changed."
    )

    if host == "smtp.sendgrid.net":
        _send_via_sendgrid_api(to_email, from_addr, subject, body, password)
        return

    port = int(os.environ.get("SMTP_PORT", 587))
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to_email

    with smtplib.SMTP(host, port, timeout=15) as server:
        server.starttls()
        server.login(user, password)
        server.sendmail(from_addr, [to_email], msg.as_string())


def _send_via_sendgrid_api(to_email: str, from_addr: str, subject: str, body: str, api_key: str) -> None:
    resp = requests.post(
        "https://api.sendgrid.com/v3/mail/send",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "personalizations": [{"to": [{"email": to_email}]}],
            "from": {"email": from_addr},
            "subject": subject,
            "content": [{"type": "text/plain", "value": body}],
        },
        timeout=15,
    )
    if resp.status_code >= 300:
        raise RuntimeError(f"SendGrid API error {resp.status_code}: {resp.text[:500]}")
