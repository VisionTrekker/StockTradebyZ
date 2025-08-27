import os
import smtplib
import logging
import argparse
from email.mime.text import MIMEText
from email.header import Header
from dotenv import load_dotenv

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

def get_results_for_date(target_date: str) -> str:
    """
    Extracts log blocks from the log file where the '交易日' matches the target_date.
    """
    try:
        with open(LOG_FILE_PATH, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        logging.warning(f"Log file not found at: {LOG_FILE_PATH}")
        return ""

    matching_blocks_text = []
    
    # Find the start indices of all result blocks
    start_indices = [i for i, line in enumerate(lines) if "============== 选股结果" in line]
    
    if not start_indices:
        logging.warning("No result blocks found in log file.")
        return ""

    # Create a list of blocks, where each block is a list of lines
    blocks = []
    for i in range(len(start_indices)):
        start = start_indices[i]
        # The end of the block is the start of the next one, or the end of the file
        end = start_indices[i+1] if i + 1 < len(start_indices) else len(lines)
        blocks.append(lines[start:end])

    # The string to search for in each block
    search_string = f"交易日: {target_date}"

    # Filter blocks that contain the target trading date
    for block in blocks:
        block_text = "".join(block)
        if search_string in block_text:
            matching_blocks_text.append(block_text)

    if not matching_blocks_text:
        logging.warning(f"No selection results found for date: {target_date}")
        return ""

    # Join all matching blocks with a newline to separate them clearly
    return "\n".join(matching_blocks_text)

def send_email(content: str, target_date: str):
    """Sends an email with the given content."""
    if not all([MAIL_HOST, MAIL_PORT, MAIL_USER, MAIL_PASS, SENDER_EMAIL, RECIPIENT_EMAIL]):
        logging.error("Email configuration is incomplete. Please check your .env file.")
        return

    if not content:
        logging.info(f"No content to send for {target_date}, skipping email.")
        return

    subject = f"每日选股报告 - {target_date}"
    message = MIMEText(content, 'plain', 'utf-8')
    message['From'] = Header(f"选股机器人 <{SENDER_EMAIL}>")
    message['To'] = Header(RECIPIENT_EMAIL)
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
    parser = argparse.ArgumentParser(description="Send stock selection results via email.")
    parser.add_argument('--date', required=True, help='The target selection date in YYYY-MM-DD format.')
    args = parser.parse_args()

    results = get_results_for_date(args.date)
    send_email(results, args.date)