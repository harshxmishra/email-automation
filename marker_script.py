import subprocess
import sys
import json
import os
import base64
import logging
from typing import Dict, Optional, List, Tuple
from PIL import Image

# Third-party imports - assuming these are installed via requirements.txt
# from langchain_google_genai import ChatGoogleGenerativeAI # Option
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
import google.generativeai as genai

# Logger setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
# TODO: Replace with environment variables or a secure config management system
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE") # Placeholder
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "YOUR_GROQ_API_KEY_HERE") # Placeholder

if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
    logger.warning("GEMINI_API_KEY is not set in environment variables. Using placeholder.")
if GROQ_API_KEY == "YOUR_GROQ_API_KEY_HERE":
    logger.warning("GROQ_API_KEY is not set in environment variables. Using placeholder.")

genai.configure(api_key=GEMINI_API_KEY)


MARKER_OUTPUT_DIR_BASE = "marker_output"
MAX_IMAGES_TO_DESCRIBE = 5 # Limit number of images sent for description to control costs/time

# --- HELPER FUNCTIONS ---

def run_marker(pdf_path: str, paper_arxiv_id: str) -> Optional[str]:
    """
    Runs marker_single to convert PDF to markdown and extract images.
    Returns the path to the directory containing extracted markdown and images, or None on failure.
    """
    if not os.path.exists(pdf_path):
        logger.error(f"PDF file not found at: {pdf_path}")
        return None

    # Create a unique output directory for this paper's marker processing
    # e.g., marker_output/2401.12345/
    specific_marker_output_dir = os.path.join(MARKER_OUTPUT_DIR_BASE, paper_arxiv_id)
    os.makedirs(specific_marker_output_dir, exist_ok=True)
    # The actual content (markdown, figures) will be in a subdirectory named after the PDF file (without extension)
    # e.g., marker_output/2401.12345/2401.12345/
    pdf_filename_stem = os.path.splitext(os.path.basename(pdf_path))[0]
    # This is where marker_single will create its subfolder named after the pdf.
    # So, the actual content will be in specific_marker_output_dir / pdf_filename_stem

    logger.info(f"Running marker_single for {pdf_path}. Output will be in a subfolder of {specific_marker_output_dir}")
    try:
        # marker_single creates a directory named after the PDF file (e.g., 'arxiv_id_stem')
        # inside the specified --output_dir.
        result = subprocess.run(
            ['marker_single', pdf_path, '--output_dir', specific_marker_output_dir, '--batch_multiplier', '1', '--max_pages', '20'], # Added some params to speed up
            capture_output=True, text=True, check=True, timeout=300 # 5 min timeout
        )
        logger.info(f"marker_single stdout: {result.stdout}")
        logger.info(f"marker_single stderr: {result.stderr}")

        # The actual content directory is specific_marker_output_dir / pdf_filename_stem
        content_directory = os.path.join(specific_marker_output_dir, pdf_filename_stem)
        if os.path.exists(content_directory):
            logger.info(f"Marker processing successful. Content in: {content_directory}")
            return content_directory # This is the directory like 'marker_output/2401.12345/2401.12345_output'
        else:
            logger.error(f"Marker output directory {content_directory} not found after running.")
            return None

    except subprocess.CalledProcessError as e:
        logger.error(f"marker_single failed for {pdf_path}: {e}")
        logger.error(f"Stderr: {e.stderr}")
        logger.error(f"Stdout: {e.stdout}")
        return None
    except subprocess.TimeoutExpired:
        logger.error(f"marker_single timed out for {pdf_path}.")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred running marker_single for {pdf_path}: {e}")
        return None

def encode_image(image_path: str) -> Optional[str]:
    """Encodes an image to base64."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except IOError as e:
        logger.error(f"Error reading image file {image_path} for encoding: {e}")
        return None

def describe_images(image_paths: List[str], groq_api_key: str) -> Dict[str, str]:
    """
    Generates one-line descriptions for a list of images using Groq API.
    """
    if not image_paths:
        return {}
    if groq_api_key == "YOUR_GROQ_API_KEY_HERE":
        logger.warning("GROQ_API_KEY is not configured. Skipping image description.")
        return {img_path: "Image description skipped (API key not configured)" for img_path in image_paths}

    img_summary_dict = {}
    try:
        model = ChatGroq(model_name="llama3-8b-8192", temperature=0.7, api_key=groq_api_key) # Updated model
    except Exception as e:
        logger.error(f"Failed to initialize Groq model for image description: {e}")
        return {img_path: f"Error initializing model: {e}" for img_path in image_paths}

    logger.info(f"Describing {len(image_paths)} images...")
    for image_path in image_paths:
        base64_image = encode_image(image_path)
        if not base64_image:
            img_summary_dict[image_path] = "Error encoding image."
            continue

        message_content = [
            {"type": "text", "text": f"Describe this technical image from a research paper in one concise sentence. This description will be used to categorize the image. Image name: {os.path.basename(image_path)}."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]
        message = HumanMessage(content=message_content)
        try:
            response = model.invoke([message])
            description = response.content.strip()
            img_summary_dict[image_path] = description
            logger.info(f"Image: {image_path}, Description: {description}")
        except Exception as e:
            logger.error(f"Error describing image {image_path} with Groq: {e}")
            img_summary_dict[image_path] = f"Error during API call: {e}"
    return img_summary_dict

def select_best_image(research_paper_summary: str, image_summaries: Dict[str, str], groq_api_key: str) -> Optional[Dict[str, str]]:
    """
    Selects the best image to display with the research paper summary using Groq API.
    Returns a dictionary like {"image_path": "path/to/image.jpg", "reason": "why it was chosen"} or None.
    """
    if not image_summaries:
        logger.info("No image summaries provided to select_best_image.")
        return None
    if groq_api_key == "YOUR_GROQ_API_KEY_HERE":
        logger.warning("GROQ_API_KEY is not configured. Skipping best image selection.")
        return {"image_path": "Skipped", "reason": "API key not configured for selection."}

    prompt = f"""Given the following summary of a research paper and a dictionary of image paths to their descriptions (from the same paper):
Research Paper Summary:
{research_paper_summary}

Image Descriptions:
{json.dumps(image_summaries, indent=2)}

Your task is to select the single best image that visually supports or is most relevant to the research paper summary.
Output a JSON object with the keys "image_path" (the full path string from the input) and "reason" (a brief explanation for your choice).
If no image is suitable or relevant, return a JSON object with "image_path": "No suitable image" and "reason": "None of the provided images were deemed relevant to the summary.".

Strictly output only the JSON object. For example:
{{
  "image_path": "marker_output/2401.0001/figure1.jpg",
  "reason": "This diagram directly illustrates the core architecture discussed in the summary."
}}
"""
    logger.info("Selecting best image using Groq...")
    try:
        llm = ChatGroq(model_name="llama3-8b-8192", temperature=0.5, api_key=groq_api_key) # Updated model
        response = llm.invoke(prompt)
        response_content = response.content.strip()
        logger.debug(f"Raw response from select_best_image LLM: {response_content}")

        # Attempt to parse as JSON
        selected_image_data = json.loads(response_content)

        # Validate that the image_path exists in the original image_summaries, unless it's "No suitable image"
        if selected_image_data.get("image_path") != "No suitable image" and \
           selected_image_data.get("image_path") not in image_summaries:
            logger.warning(f"LLM selected image_path '{selected_image_data.get('image_path')}' which is not in the provided list. Defaulting to no image.")
            return {"image_path": "No suitable image", "reason": "LLM chose an invalid image path."}

        logger.info(f"Selected image data: {selected_image_data}")
        return selected_image_data

    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON response from select_best_image LLM. Response: {response_content}")
        # Fallback: try to find the path if it's just a string response (less robust)
        for path in image_summaries.keys():
            if path in response_content: # very basic check
                 return {"image_path": path, "reason": "Inferred from LLM response (JSON parse failed)."}
        return {"image_path": "Error", "reason": "Failed to parse LLM response for image selection."}
    except Exception as e:
        logger.error(f"Error during select_best_image with Groq: {e}")
        return {"image_path": "Error", "reason": f"API call failed: {e}"}


def summarize_paper_with_gemini(title: str, authors: str, pdf_path: str, gemini_api_key: str) -> Optional[Tuple[str, str]]:
    """
    Summarizes a research paper using the Gemini API and a prompt template.
    Returns a tuple (summary, category) or None on failure.
    """
    if gemini_api_key == "YOUR_GEMINI_API_KEY_HERE":
        logger.warning("GEMINI_API_KEY is not configured. Skipping paper summarization.")
        return None, None

    try:
        # Check if prompt.md exists
        if not os.path.exists("prompt.md"):
            logger.error("prompt.md file not found. Cannot generate summary.")
            return None, None
        with open("prompt.md", "r") as f:
            prompt_template = f.read()
    except IOError as e:
        logger.error(f"Error reading prompt.md: {e}")
        return None, None

    try:
        model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest") # Using recommended latest flash
    except Exception as e:
        logger.error(f"Failed to initialize Gemini model: {e}")
        return None, None

    logger.info(f"Summarizing paper: {title} using Gemini...")
    try:
        # PDF upload to Gemini (consider if marker's text output is better for Gemini)
        # For now, using direct PDF upload as in original script.
        logger.info(f"Uploading PDF {pdf_path} to Gemini...")
        pdf_file_upload = genai.upload_file(path=pdf_path, display_name=os.path.basename(pdf_path))
        logger.info(f"PDF uploaded successfully: {pdf_file_upload.name}")

        prompt = prompt_template.replace("{title}", title).replace("{authors}", authors)

        # Generate content
        response = model.generate_content([pdf_file_upload, prompt], request_options={'timeout': 300}) # 5 min timeout
        response_text = response.text.strip()
        logger.debug(f"Raw response from Gemini: {response_text}")

        # Clean potential markdown ```json ... ```
        if response_text.startswith("```json"):
            response_text = response_text.replace("```json", "").replace("```", "").strip()

        response_data = json.loads(response_text)
        summary = response_data.get("summary")
        category = response_data.get("category")

        if not summary or not category:
            logger.error(f"Gemini response missing 'summary' or 'category'. Data: {response_data}")
            return None, None

        logger.info(f"Paper '{title}' summarized. Category: {category}")
        # Delete the uploaded file from Gemini to manage storage
        try:
            genai.delete_file(pdf_file_upload.name)
            logger.info(f"Deleted uploaded file {pdf_file_upload.name} from Gemini.")
        except Exception as e:
            logger.warning(f"Could not delete file {pdf_file_upload.name} from Gemini: {e}")

        return summary, category

    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON response from Gemini. Response: {response_text}")
        return None, None
    except Exception as e:
        logger.error(f"Error during Gemini summarization for '{title}': {e}", exc_info=True)
        # Attempt to delete file even if summarization failed
        if 'pdf_file_upload' in locals() and hasattr(pdf_file_upload, 'name'):
            try:
                genai.delete_file(pdf_file_upload.name)
                logger.info(f"Deleted uploaded file {pdf_file_upload.name} from Gemini after error.")
            except Exception as del_e:
                logger.warning(f"Could not delete file {pdf_file_upload.name} from Gemini after error: {del_e}")
        return None, None


def process_paper_data(paper_info: Dict) -> Optional[Dict]:
    """
    Main processing function for a single paper.
    Takes paper_info dictionary (expected to have 'pdf_path', 'arxiv_id', 'title', 'authors').
    Returns a dictionary with summary, category, and image details, or None on failure.
    """
    pdf_path = paper_info.get("pdf_path")
    arxiv_id = paper_info.get("arxiv_id")
    title = paper_info.get("title", "N/A")
    authors = paper_info.get("authors", "N/A")

    if not all([pdf_path, arxiv_id]):
        logger.error("Missing pdf_path or arxiv_id in paper_info.")
        return None

    # 1. Run Marker to get text and images
    # This path will be like 'marker_output/arxiv_id_stem/arxiv_id_stem_output'
    marker_content_dir = run_marker(pdf_path, arxiv_id)
    if not marker_content_dir:
        logger.error(f"Marker processing failed for {arxiv_id}. Cannot proceed.")
        return None

    # 2. Summarize paper using Gemini
    # The original script used PDF for Gemini. We can continue this or use marker's MD output.
    # Sticking to PDF for now to match original logic closer for summarization quality.
    summary, category = summarize_paper_with_gemini(title, authors, pdf_path, GEMINI_API_KEY)
    if not summary or not category:
        logger.error(f"Failed to get summary or category for {title} ({arxiv_id}).")
        # Proceeding without summary for now, might still be ableto pick an image if needed,
        # but likely the process should fail or return partial data.
        # For now, let's make summary crucial.
        return {
            "arxiv_id": arxiv_id,
            "title": title,
            "summary": "Error: Failed to generate summary.",
            "category": "Error",
            "selected_image_path": None,
            "selected_image_reason": None,
            "status": "Failed - Summarization error"
        }


    # 3. List and Describe Images from Marker output
    # Images are typically in a 'figures' subdirectory within the marker_content_dir
    figures_dir = os.path.join(marker_content_dir, "figures")
    image_paths = []
    if os.path.exists(figures_dir):
        image_paths = [
            os.path.join(figures_dir, f)
            for f in os.listdir(figures_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ][:MAX_IMAGES_TO_DESCRIBE] # Limit number of images
        logger.info(f"Found {len(image_paths)} images in {figures_dir} (capped at {MAX_IMAGES_TO_DESCRIBE}).")
    else:
        logger.info(f"No 'figures' directory found in {marker_content_dir}. No images to process.")

    image_descriptions = {}
    if image_paths:
        image_descriptions = describe_images(image_paths, GROQ_API_KEY)

    # 4. Select Best Image based on summary and image descriptions
    selected_image_info = None
    if summary and image_descriptions: # Only select if we have a summary and descriptions
        selected_image_info = select_best_image(summary, image_descriptions, GROQ_API_KEY)
    elif not image_descriptions:
        logger.info("No image descriptions available, skipping best image selection.")
    elif not summary:
         logger.info("No summary available, skipping best image selection.")


    return {
        "arxiv_id": arxiv_id,
        "title": title,
        "authors": authors,
        "summary": summary,
        "category": category,
        "selected_image_path": selected_image_info.get("image_path") if selected_image_info else None,
        "selected_image_reason": selected_image_info.get("reason") if selected_image_info else None,
        "all_image_descriptions": image_descriptions, # For debugging or other uses
        "marker_content_path": marker_content_dir, # Path to where md and figures are
        "status": "Success"
    }

if __name__ == "__main__":
    # This is for testing the script directly.
    # In the automation, main_automation.py will call process_paper_data.
    logger.info("Starting marker_script.py for direct testing.")

    # --- Dummy paper_info for testing ---
    # In a real run, this would come from hf_script.py
    # Ensure you have a PDF file at 'temp_pdfs/test_paper.pdf' for this test.
    os.makedirs("temp_pdfs", exist_ok=True)
    os.makedirs(MARKER_OUTPUT_DIR_BASE, exist_ok=True)

    # Create a dummy PDF if it doesn't exist for testing
    dummy_pdf_path = "temp_pdfs/2401.00001.pdf"
    if not os.path.exists(dummy_pdf_path):
        try:
            with open(dummy_pdf_path, "wb") as f:
                # A very small valid PDF file content (basically empty)
                f.write(b"%PDF-1.0\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj 2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj 3 0 obj<</Type/Page/MediaBox[0 0 3 3]>>endobj\nxref\n0 4\n0000000000 65535 f\n0000000010 00000 n\n0000000058 00000 n\n0000000115 00000 n\ntrailer<</Size 4/Root 1 0 R>>\nstartxref\n142\n%%EOF")
            logger.info(f"Created dummy PDF: {dummy_pdf_path}")
        except Exception as e:
            logger.error(f"Could not create dummy PDF: {e}")
            sys.exit(1)

    # Create a dummy prompt.md if it doesn't exist
    if not os.path.exists("prompt.md"):
        try:
            with open("prompt.md", "w") as f:
                f.write("""{
"category": "Other",
"summary": "### Overall Summary\n* This is a dummy summary for {title} by {authors} because prompt.md was missing."
}""")
            logger.info("Created dummy prompt.md for testing.")
        except Exception as e:
            logger.error(f"Could not create dummy prompt.md: {e}")
            # Not exiting, summarization will fail gracefully

    test_paper_info = {
        "pdf_path": dummy_pdf_path, # Replace with a real PDF from hf_script for better testing
        "arxiv_id": "2401.00001", # Should match the PDF filename stem for marker
        "title": "Test Paper for Marker Script",
        "authors": "J. Doe, A. Nonymous",
        # These would normally come from hf_script.py
        "link": "http://arxiv.org/abs/2401.00001",
        "upvotes": 10
    }

    # Ensure API keys are set as environment variables for this test to fully work
    # export GEMINI_API_KEY="your_key"
    # export GROQ_API_KEY="your_key"
    if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE" or GROQ_API_KEY == "YOUR_GROQ_API_KEY_HERE":
        logger.warning("API keys are not set. Summarization and image description might be skipped or fail.")
        logger.warning("Set GEMINI_API_KEY and GROQ_API_KEY environment variables to test fully.")
        # Create a dummy marker output structure for partial testing if keys are missing
        dummy_marker_content_path = os.path.join(MARKER_OUTPUT_DIR_BASE, test_paper_info["arxiv_id"], f"{test_paper_info['arxiv_id']}_output")
        dummy_figures_path = os.path.join(dummy_marker_content_path, "figures")
        os.makedirs(dummy_figures_path, exist_ok=True)
        with open(os.path.join(dummy_marker_content_path, f"{test_paper_info['arxiv_id']}.md"), "w") as f:
            f.write("# Dummy Markdown\nContent of the paper.")
        # Create a dummy image
        try:
            from PIL import Image as PillowImage
            img = PillowImage.new('RGB', (60, 30), color = 'red')
            img.save(os.path.join(dummy_figures_path, 'dummy_fig.png'))
            logger.info(f"Created dummy figure in {dummy_figures_path}")
        except Exception as e:
            logger.error(f"Could not create dummy image for testing: {e}")


    logger.info(f"Processing test paper: {test_paper_info['title']}")
    result = process_paper_data(test_paper_info)

    if result:
        logger.info("\n--- Processing Result ---")
        logger.info(f"Title: {result.get('title')}")
        logger.info(f"Arxiv ID: {result.get('arxiv_id')}")
        logger.info(f"Category: {result.get('category')}")
        logger.info(f"Summary: \n{result.get('summary')}")
        logger.info(f"Selected Image Path: {result.get('selected_image_path')}")
        logger.info(f"Selected Image Reason: {result.get('selected_image_reason')}")
        logger.info(f"Marker Content Path: {result.get('marker_content_path')}")
        logger.info(f"Status: {result.get('status')}")
        # logger.info(f"All Image Descriptions: {json.dumps(result.get('all_image_descriptions'), indent=2)}")
        logger.info("--- End of Result ---")
    else:
        logger.error("Processing failed for the test paper.")

    # Clean up dummy files for next run? Optional.
    # if os.path.exists(dummy_pdf_path): os.remove(dummy_pdf_path)
    # if os.path.exists("prompt.md_dummy"): os.rename("prompt.md_dummy", "prompt.md") # if backed up
    # import shutil
    # if os.path.exists(MARKER_OUTPUT_DIR_BASE): shutil.rmtree(MARKER_OUTPUT_DIR_BASE)
    # logger.info("Cleaned up dummy files and directories.")
else:
    logger.info("marker_script.py loaded as a module.")
