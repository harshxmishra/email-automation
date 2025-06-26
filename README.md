# Email Automation for Hugging Face Papers

This project automates the process of fetching the top daily research paper from Hugging Face Papers, summarizing it, selecting a relevant image, and preparing this information for (simulated) email dispatch. The automation is scheduled to run three times a day.

## Features

-   Fetches the latest papers from Hugging Face.
-   Identifies the top paper based on upvotes, avoiding duplicates within the same day.
-   Uses `marker-pdf` to convert the paper's PDF to Markdown and extract images.
-   Utilizes LLMs (Gemini and Groq) to:
    -   Summarize the research paper.
    -   Generate descriptions for extracted images.
    -   Select the most relevant image for the summary.
-   Schedules the entire process to run at 10:30 AM, 2:00 PM, and 5:00 PM daily.
-   Logs actions and sends email notifications with the paper details and selected image.

## Project Structure

-   `main_automation.py`: The main orchestrator script that runs the scheduled tasks.
-   `hf_script.py`: Responsible for fetching paper information from Hugging Face, selecting the top new paper, and downloading its PDF. Manages `processed_papers.txt` for duplicate tracking.
-   `marker_script.py`: Handles the processing of a single paper: PDF to Markdown conversion, image extraction, summarization, and relevant image selection using LLMs.
-   `prompt.md`: The template used by `marker_script.py` to instruct the Gemini LLM on how to summarize papers.
-   `requirements.txt`: Lists all Python dependencies.
-   `processed_papers.txt`: (Auto-generated) Tracks IDs of papers processed during the current day to prevent duplicates. Cleared daily.
-   `last_reset_date.txt`: (Auto-generated) Stores the date when `processed_papers.txt` was last cleared.
-   `temp_pdfs/`: (Auto-generated) Directory where downloaded PDFs are stored temporarily.
-   `marker_output/`: (Auto-generated) Directory where `marker-pdf` stores its output (markdown files, images).

## Setup and Configuration

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Create a Python virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up Environment Variables:**
    This project requires several environment variables to be set:

    *   **LLM API Keys:**
        ```bash
        export GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
        export GROQ_API_KEY="YOUR_GROQ_API_KEY"
        ```
        Replace with your actual keys from Google AI Studio and GroqCloud.

    *   **Email Configuration:**
        ```bash
        export EMAIL_HOST="your_smtp_server.com"
        export EMAIL_PORT="587"  # Or 465 for SSL, adjust as needed
        export EMAIL_HOST_USER="your_sender_email@example.com"
        export EMAIL_HOST_PASSWORD="your_email_password_or_app_token"
        export EMAIL_RECIPIENTS="recipient1@example.com,recipient2@example.com"
        ```
        -   `EMAIL_HOST`: SMTP server address (e.g., `smtp.gmail.com`).
        -   `EMAIL_PORT`: SMTP server port (e.g., `587` for TLS, `465` for SSL).
        -   `EMAIL_HOST_USER`: The email address used for sending.
        -   `EMAIL_HOST_PASSWORD`: The password for the sender email address. **For services like Gmail, if you have 2-Factor Authentication enabled, you'll likely need to generate an "App Password" to use here.**
        -   `EMAIL_RECIPIENTS`: A comma-separated list of email addresses to send the notification to.

    If you are on Windows, use `set VARIABLE_NAME="VALUE"` (e.g., `set GEMINI_API_KEY="YOUR_KEY"`) in your command prompt or configure them through the System Properties.

## Running the Automation

Once the setup is complete and API keys are configured, you can run the main automation script:

```bash
python main_automation.py
```

The script will start the scheduler and run the tasks at the predefined times (10:30 AM, 2:00 PM, 5:00 PM). It will log its actions to the console.

**To test a single run of the core task immediately (outside of the schedule):**
You can modify `main_automation.py` temporarily by adding a direct call to `run_ મુખ્ય_task()` within the `if __name__ == "__main__":` block before `main()` is called, or by reducing the schedule times for quicker testing. Remember to revert these changes for normal operation.

Alternatively, you can test individual scripts:
-   `python hf_script.py`: This will attempt to fetch and print details of the current top new paper.
-   `python marker_script.py`: This will run with its built-in test data (a dummy PDF and prompt), useful for checking if `marker-pdf` and LLM interactions are working.

## Notes

-   **Email Sending**: The system now implements actual email sending. Ensure your `EMAIL_*` environment variables are correctly configured. If using Gmail with 2FA, remember to use an App Password.
-   The `marker-pdf` tool can be resource-intensive. Ensure your environment has sufficient CPU/memory.
-   LLM API calls can incur costs. Monitor your usage on the respective platforms (Google AI Studio for Gemini, GroqCloud for Groq).
-   The `marker_single` command in `marker_script.py` has parameters like `--batch_multiplier` and `--max_pages` that can be adjusted if processing takes too long or if you encounter issues with specific PDFs.
-   Ensure that `marker_single` is correctly installed and accessible in your system's PATH or Python environment if you installed `marker-pdf` globally. If installed in a venv, `subprocess.run` should find it.