import schedule
import time
import logging
import os
from datetime import datetime

# Import functions from other scripts
from hf_script import get_top_new_paper, PROCESSED_PAPERS_FILE
from marker_script import process_paper_data, GEMINI_API_KEY, GROQ_API_KEY # Import API key constants for checking

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import markdown # For converting summary to HTML

# Logger setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [MAIN] - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
SCHEDULE_TIMES = ["10:30", "14:00", "17:00"] # 24-hour format

# File to store the date of the last reset for processed_papers.txt
LAST_RESET_DATE_FILE = "last_reset_date.txt"

# --- Environment Variable Checks ---
def check_environment_variables():
    """Checks if all necessary environment variables seem to be configured."""
    all_keys_ok = True
    # API Keys for services
    if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        logger.warning("GEMINI_API_KEY is not set in environment variables. marker_script may not function fully.")
        all_keys_ok = False
    if GROQ_API_KEY == "YOUR_GROQ_API_KEY_HERE":
        logger.warning("GROQ_API_KEY is not set in environment variables. marker_script may not function fully.")
        all_keys_ok = False

    # Email Credentials
    email_vars = {
        "EMAIL_HOST": os.environ.get("EMAIL_HOST"),
        "EMAIL_PORT": os.environ.get("EMAIL_PORT"),
        "EMAIL_HOST_USER": os.environ.get("EMAIL_HOST_USER"),
        "EMAIL_HOST_PASSWORD": os.environ.get("EMAIL_HOST_PASSWORD"),
        "EMAIL_RECIPIENTS": os.environ.get("EMAIL_RECIPIENTS")
    }
    for var_name, value in email_vars.items():
        if not value:
            logger.warning(f"{var_name} is not set in environment variables. Email functionality will be disabled.")
            all_keys_ok = False

    if all_keys_ok:
        logger.info("All checked environment variables (GEMINI_API_KEY, GROQ_API_KEY, and Email settings) appear to be configured.")
    else:
        logger.error("One or more critical environment variables are missing. Please check the logs.")
    return all_keys_ok

# --- Email Sending Function ---
def send_email(subject: str, html_body: str, image_path: Optional[str]) -> bool:
    """
    Sends an email with the given subject, HTML body, and optionally an attached image.
    Reads email configuration from environment variables.
    """
    email_host = os.environ.get("EMAIL_HOST")
    email_port_str = os.environ.get("EMAIL_PORT", "587") # Default to 587 if not set
    email_host_user = os.environ.get("EMAIL_HOST_USER")
    email_host_password = os.environ.get("EMAIL_HOST_PASSWORD")
    email_recipients_str = os.environ.get("EMAIL_RECIPIENTS")

    if not all([email_host, email_port_str, email_host_user, email_host_password, email_recipients_str]):
        logger.error("Email configuration is incomplete. Cannot send email. Please set EMAIL_HOST, EMAIL_PORT, EMAIL_HOST_USER, EMAIL_HOST_PASSWORD, and EMAIL_RECIPIENTS environment variables.")
        return False

    try:
        email_port = int(email_port_str)
    except ValueError:
        logger.error(f"Invalid EMAIL_PORT: {email_port_str}. Must be an integer.")
        return False

    recipients = [email.strip() for email in email_recipients_str.split(',') if email.strip()]
    if not recipients:
        logger.error("No recipients specified in EMAIL_RECIPIENTS.")
        return False

    msg = MIMEMultipart('related')
    msg['Subject'] = subject
    msg['From'] = email_host_user
    msg['To'] = ", ".join(recipients)

    # Attach HTML body
    msg.attach(MIMEText(html_body, 'html'))

    # Attach image if path is provided and image exists
    if image_path and os.path.exists(image_path):
        try:
            with open(image_path, 'rb') as img_file:
                img = MIMEImage(img_file.read())
                img.add_header('Content-ID', '<paper_image>') # Referenced by <img src="cid:paper_image">
                img.add_header('Content-Disposition', 'inline', filename=os.path.basename(image_path))
                msg.attach(img)
            logger.info(f"Attached image {image_path} to email.")
        except Exception as e:
            logger.error(f"Failed to attach image {image_path}: {e}")
    elif image_path:
        logger.warning(f"Image path {image_path} provided but file does not exist. Email will be sent without image.")


    try:
        logger.info(f"Connecting to SMTP server {email_host}:{email_port}...")
        with smtplib.SMTP(email_host, email_port) as server:
            server.ehlo()
            server.starttls() # Enable TLS
            server.ehlo()
            server.login(email_host_user, email_host_password)
            server.sendmail(email_host_user, recipients, msg.as_string())
        logger.info(f"Email sent successfully to: {', '.join(recipients)}")
        return True
    except smtplib.SMTPAuthenticationError as e:
        logger.error(f"SMTP Authentication Error: {e}. Check username/password or app-specific password settings.")
        return False
    except Exception as e:
        logger.error(f"Failed to send email: {e}", exc_info=True)
        return False

def clear_processed_papers_if_new_day():
    """Clears the processed_papers.txt file if it's a new day."""
    today_str = datetime.now().strftime("%Y-%m-%d")
    last_reset_date = ""

    if os.path.exists(LAST_RESET_DATE_FILE):
        try:
            with open(LAST_RESET_DATE_FILE, "r") as f:
                last_reset_date = f.read().strip()
        except IOError as e:
            logger.error(f"Could not read {LAST_RESET_DATE_FILE}: {e}")
            # Proceed as if it's a new day to be safe
            last_reset_date = ""


    if last_reset_date != today_str:
        logger.info(f"New day detected (or first run). Last reset was on {last_reset_date}, today is {today_str}.")
        if os.path.exists(PROCESSED_PAPERS_FILE):
            try:
                os.remove(PROCESSED_PAPERS_FILE)
                logger.info(f"Cleared {PROCESSED_PAPERS_FILE} for the new day.")
            except OSError as e:
                logger.error(f"Error clearing {PROCESSED_PAPERS_FILE}: {e}")
                # If we can't clear it, we might send duplicates. Critical error.
                return False # Indicate failure

        try:
            with open(LAST_RESET_DATE_FILE, "w") as f:
                f.write(today_str)
            logger.info(f"Updated {LAST_RESET_DATE_FILE} to {today_str}.")
        except IOError as e:
            logger.error(f"Could not write to {LAST_RESET_DATE_FILE}: {e}")
            # This is not ideal, as it might lead to multiple clears on the same day if script restarts
            return False # Indicate failure
    else:
        logger.info(f"Already ran today ({today_str}). {PROCESSED_PAPERS_FILE} will not be cleared yet.")
    return True


def run_ મુખ્ય_task(): # Renamed to avoid conflict if we want to use "run_daily_task" for something else
    """
    The main task to be scheduled. Fetches top paper, processes it, and logs info.
    """
    logger.info("Starting scheduled task: run_ મુખ્ય_task")

    if not clear_processed_papers_if_new_day():
        logger.error("Failed to manage processed papers log for the new day. Skipping task to avoid issues.")
        return

    # 1. Get the top new paper
    logger.info("Fetching top new paper from Hugging Face...")
    top_paper_details = get_top_new_paper() # From hf_script.py

    if not top_paper_details:
        logger.info("No new top paper found to process at this time.")
        return

    logger.info(f"New top paper found: {top_paper_details.get('title')} (ID: {top_paper_details.get('arxiv_id')})")
    logger.info(f"PDF Path: {top_paper_details.get('pdf_path')}")

    # 2. Process the paper data (summary, image selection)
    logger.info(f"Processing paper data for: {top_paper_details.get('title')}...")
    # Ensure API keys are loaded in marker_script's context if they were changed post-import
    # This is generally handled by marker_script itself reading env vars at its load time or function call.

    processed_data = process_paper_data(top_paper_details) # From marker_script.py

    if not processed_data or processed_data.get("status") != "Success":
        logger.error(f"Failed to process paper data for {top_paper_details.get('title')}. Result: {processed_data}")
        return

    logger.info(f"Successfully processed paper: {processed_data.get('title')}")
    logger.info(f"  Category: {processed_data.get('category')}")
    logger.info(f"  Summary: \n{processed_data.get('summary')}")
    logger.info(f"  Selected Image Path: {processed_data.get('selected_image_path')}")
    logger.info(f"  Selected Image Reason: {processed_data.get('selected_image_reason')}")

    # 3. Prepare and Send Email
    email_subject = f"Top AI Paper: {processed_data.get('title')}"

    # Convert summary from Markdown to HTML
    summary_html = ""
    if processed_data.get('summary'):
        try:
            summary_html = markdown.markdown(processed_data.get('summary'), extensions=['extra', 'nl2br'])
        except Exception as e:
            logger.error(f"Error converting summary to HTML: {e}")
            summary_html = f"<p>Error displaying summary: {processed_data.get('summary')}</p>" # Fallback
    else:
        summary_html = "<p>No summary available.</p>"

    # Construct HTML body
    # Using f-string and then .format for values that might contain braces or need careful escaping.
    # Simpler f-string approach here since content is mostly controlled.
    html_body_template = """
    <html>
      <head>
        <style>
          body {{ font-family: sans-serif; line-height: 1.6; margin: 20px; }}
          h2 {{ color: #333; }}
          h3 {{ color: #555; margin-top: 30px; }}
          p, div {{ margin-bottom: 15px; }}
          .paper-details p {{ margin-bottom: 5px; }}
          .summary-content {{ background-color: #f9f9f9; border-left: 4px solid #007bff; padding: 15px; }}
          img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; margin-top: 10px; }}
          .image-reason {{ font-style: italic; color: #777; font-size: 0.9em;}}
        </style>
      </head>
      <body>
        <h2>{title}</h2>
        <div class="paper-details">
            <p><strong>Authors:</strong> {authors}</p>
            <p><strong>Category:</strong> {category}</p>
            <p><strong>ArXiv ID:</strong> <a href="https://arxiv.org/abs/{arxiv_id}">{arxiv_id}</a></p>
            <p><strong>Original Link:</strong> <a href="{paper_link}">{paper_link}</a></p>
        </div>
    """

    selected_image_path = processed_data.get('selected_image_path')
    if selected_image_path and os.path.exists(selected_image_path):
        html_body_template += """
        <h3>Visual Highlight</h3>
        <p><img src="cid:paper_image" alt="Selected Paper Image"></p>
        <p class="image-reason"><em>Reason: {image_reason}</em></p>
        """
    else:
        html_body_template += "<p><em>No suitable image was selected or found for this paper.</em></p>"

    html_body_template += """
        <h3>Summary</h3>
        <div class="summary-content">
          {summary_html}
        </div>
        <hr>
        <p><small>Automated Paper Notification Service</small></p>
      </body>
    </html>
    """

    html_body = html_body_template.format(
        title=processed_data.get('title', 'N/A'),
        authors=top_paper_details.get('authors', 'N/A'), # Get authors from original hf_script data
        category=processed_data.get('category', 'N/A'),
        arxiv_id=top_paper_details.get('arxiv_id', 'N/A'), # Get arxiv_id from original hf_script data
        paper_link=top_paper_details.get('link', '#'), # Get link from original hf_script data
        image_reason=processed_data.get('selected_image_reason', ''),
        summary_html=summary_html
    )

    email_sent = send_email(email_subject, html_body, selected_image_path)

    if email_sent:
        logger.info("Email sent successfully for paper: %s", processed_data.get('title'))
    else:
        logger.error("Failed to send email for paper: %s", processed_data.get('title'))
        # Log the content that would have been sent for debugging if email fails
        logger.debug(f"\n--- FAILED EMAIL CONTENT ---")
        logger.debug(f"To: {os.environ.get('EMAIL_RECIPIENTS')}")
        logger.debug(f"Subject: {email_subject}")
        logger.debug(f"Body (HTML):\n{html_body[:500]}...") # Log beginning of body
        logger.debug("--- END OF FAILED EMAIL CONTENT ---\n")

    logger.info("run_ મુખ્ય_task completed.")


def main():
    logger.info("Starting Main Automation Script.")

    if not check_environment_variables(): # Updated function name
        logger.error("One or more critical environment variables are not configured correctly. Please review logs. Exiting.")
        return

    # Initial check/clear of processed papers log
    if not clear_processed_papers_if_new_day():
        logger.critical("Could not manage processed papers log on startup. Exiting to prevent duplicates.")
        return

    # Schedule the tasks
    for t in SCHEDULE_TIMES:
        logger.info(f"Scheduling main task daily at {t}")
        schedule.every().day.at(t).do(run_ મુખ્ય_task)

    logger.info("Scheduler started. Waiting for scheduled tasks...")
    logger.info(f"Next run times: {[job.next_run.strftime('%Y-%m-%d %H:%M:%S') for job in schedule.jobs if job.next_run]}")


    # Keep the script running to execute scheduled jobs
    try:
        while True:
            schedule.run_pending()
            time.sleep(60) # Check every minute
            # Log next run time periodically for visibility
            # if datetime.now().minute == 0: # Log every hour
            #    logger.debug(f"Still running. Next jobs: {[job.next_run for job in schedule.jobs]}")
    except KeyboardInterrupt:
        logger.info("Script interrupted by user. Exiting.")
    except Exception as e:
        logger.error(f"An unexpected error occurred in the main loop: {e}", exc_info=True)
    finally:
        logger.info("Main Automation Script finished.")

if __name__ == "__main__":
    # Create necessary directories if they don't exist, for robustness
    os.makedirs("temp_pdfs", exist_ok=True)
    os.makedirs("marker_output", exist_ok=True) # Base for marker outputs

    main()
