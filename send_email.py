
import os
import smtplib
import logging
from email.mime.text import MIMEText
from email.header import Header
from dotenv import load_dotenv
from datetime import date

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
load_dotenv()

# --- Environment Variables ---
MAIL_HOST = os.getenv("MAIL_HOST")
MAIL_PORT = os.getenv("MAIL_PORT")
MAIL_USER = os.getenv("MAIL_USER")
MAIL_PASS = os.getenv("MAIL_PASS")
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
RECIPIENT_EMAIL = os.getenv("RECIPIENT_EMAIL")

LOG_FILE_PATH = "select_results.log"

def get_latest_selection_results() -> str:
    """Extracts the most recent selection results from the log file."""
    try:
        with open(LOG_FILE_PATH, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        logging.warning(f"Log file not found at: {LOG_FILE_PATH}")
        return ""

    # Find the start of the last selection result block
    start_indices = [i for i, line in enumerate(lines) if "============== 选股结果" in line]
    if not start_indices:
        logging.warning("No selection results found in the log file.")
        return ""

    last_result_block = lines[start_indices[-1]:]
    return "".join(last_result_block)

def send_email(content: str):
    """Sends an email with the given content."""
    if not all([MAIL_HOST, MAIL_PORT, MAIL_USER, MAIL_PASS, SENDER_EMAIL, RECIPIENT_EMAIL]):
        logging.error("Email configuration is incomplete. Please check your .env file.")
        return

    if not content:
        logging.info("No content to send, skipping email.")
        return

    subject = f"每日选股报告 - {date.today().strftime('%Y-%m-%d')}"
    message = MIMEText(content, 'plain', 'utf-8')
    message['From'] = Header(f"选股机器人 <{SENDER_EMAIL}>", 'utf-8')
    message['To'] = Header(RECIPIENT_EMAIL, 'utf-8')
    message['Subject'] = Header(subject, 'utf-8')

    try:
        logging.info(f"Connecting to SMTP server at {MAIL_HOST}:{MAIL_PORT}...")
        with smtplib.SMTP_SSL(MAIL_HOST, int(MAIL_PORT)) as server:
            server.login(MAIL_USER, MAIL_PASS)
            server.sendmail(SENDER_EMAIL, [RECIPIENT_EMAIL], message.as_string())
        logging.info(f"Email sent successfully to {RECIPIENT_EMAIL}")
    except Exception as e:
        logging.error(f"Failed to send email: {e}")

if __name__ == "__main__":
    results = get_latest_selection_results()
    send_email(results)
