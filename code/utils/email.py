import os
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

logger = logging.getLogger(__name__)


def send_password_reset_email(recipient_email: str, recipient_name: str, reset_link: str) -> bool:
    """Send a password reset email via SMTP. Falls back to console log if SMTP credentials are not set."""
    smtp_host = os.environ.get("SMTP_HOST", "").strip()
    smtp_port = int(os.environ.get("SMTP_PORT", "587") or "587")
    smtp_user = os.environ.get("SMTP_USER", "").strip()
    smtp_pass = os.environ.get("SMTP_PASS", "").strip()
    smtp_from = os.environ.get("SMTP_FROM_EMAIL", "no-reply@x-sense.ai").strip()

    # Always log to console / logger for developer convenience & local testing
    logger.info("======================================================================")
    logger.info("PASSWORD RESET LINK FOR %s (%s):", recipient_name, recipient_email)
    logger.info("%s", reset_link)
    logger.info("======================================================================")
    print(f"\n[DEV MAIL] Password Reset Link for {recipient_email}: {reset_link}\n", flush=True)

    if not smtp_host or not smtp_user or not smtp_pass:
        logger.info("SMTP credentials not fully configured in .env. Using DEV console log fallback.")
        return True

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = "Reset Your Password - X-Sense"
        msg["From"] = smtp_from
        msg["To"] = recipient_email

        text_content = (
            f"Hello {recipient_name},\n\n"
            "You requested a password reset for your X-Sense account.\n"
            f"Please click the following link to reset your password:\n{reset_link}\n\n"
            "If you did not request this reset, please ignore this email.\nThis link will expire in 30 minutes.\n"
        )

        html_content = f"""
        <html>
          <body style="font-family: Arial, sans-serif; background-color: #0f172a; color: #f8fafc; padding: 40px 20px;">
            <div style="max-width: 500px; margin: 0 auto; background-color: #1e293b; padding: 30px; border-radius: 12px; border: 1px solid #334155;">
              <h2 style="color: #38bdf8; margin-top: 0;">Password Reset Request</h2>
              <p style="color: #cbd5e1; font-size: 16px;">Hello <strong>{recipient_name}</strong>,</p>
              <p style="color: #cbd5e1; font-size: 16px; line-height: 1.5;">
                We received a request to reset the password for your X-Sense account. Click the button below to set a new password:
              </p>
              <div style="text-align: center; margin: 30px 0;">
                <a href="{reset_link}" style="background: linear-gradient(135deg, #2563eb, #3b82f6); color: #ffffff; text-decoration: none; padding: 14px 28px; border-radius: 8px; font-weight: bold; font-size: 16px; display: inline-block;">
                  Reset Password
                </a>
              </div>
              <p style="color: #94a3b8; font-size: 14px;">
                Or copy and paste this link into your browser:<br>
                <a href="{reset_link}" style="color: #38bdf8; word-break: break-all;">{reset_link}</a>
              </p>
              <hr style="border: none; border-top: 1px solid #334155; margin: 25px 0;">
              <p style="color: #64748b; font-size: 12px; margin-bottom: 0;">
                If you did not request a password reset, you can safely ignore this email. Only someone with access to this email can reset your password. This link expires in 30 minutes.
              </p>
            </div>
          </body>
        </html>
        """

        part1 = MIMEText(text_content, "plain")
        part2 = MIMEText(html_content, "html")
        msg.attach(part1)
        msg.attach(part2)

        with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as server:
            server.ehlo()
            if smtp_port in (587, 25):
                server.starttls()
                server.ehlo()
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_from, recipient_email, msg.as_string())

        logger.info("Successfully sent password reset email to %s", recipient_email)
        return True
    except Exception as exc:
        logger.error("Failed to send SMTP password reset email to %s: %s", recipient_email, exc)
        return False
