# Standard library imports
import json
import os
import re
from datetime import datetime
from typing import List, Dict, Optional
import time

# Third party imports

import requests
from bs4 import BeautifulSoup, Tag

# Logger setup
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

HF_URL = "https://huggingface.co/papers/"

# --- download_pdf function remains the same ---
def download_pdf(arxiv_id: str, save_path: str) -> bool:
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    logger.info(f"Attempting to download PDF from: {url}")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        # time.sleep(0.5) # Optional delay
        response = requests.get(url, timeout=45, headers=headers)
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "")
        if "application/pdf" in content_type or "application/octet-stream" in content_type:
            with open(save_path, "wb") as f:
                f.write(response.content)
            # logger.debug(f"Successfully downloaded PDF to: {save_path}") # DEBUG level
            return True
        else:
            logger.warning(f"URL did not return a PDF content type ({content_type}): {url}")
            if "text/html" in content_type:
                 logger.warning(f"  -> Received HTML content instead of PDF for {arxiv_id}.")
            return False
    except requests.exceptions.Timeout:
         logger.error(f"Timeout error downloading PDF {arxiv_id} from {url}")
         return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error downloading PDF {arxiv_id}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during PDF download for {arxiv_id}: {e}")
        return False

def pull_hf_daily() -> None:
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        logger.info(f"Fetching URL: {HF_URL}")
        response = requests.get(HF_URL, timeout=20, headers=headers)
        response.raise_for_status()
        logger.info("Successfully fetched page content.")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch Hugging Face papers page: {e}")
        return

    try:
        soup = BeautifulSoup(response.content, "lxml")
    except Exception as e:
        logger.warning(f"lxml parser failed ({e}), falling back to html.parser")
        soup = BeautifulSoup(response.content, "html.parser")

    # --- Extract Upvotes and other data from Embedded JSON ---
    paper_data_from_json = {}
    try:
        # Find the Svelte hydrater div that contains the paper data
        hydrater_div = soup.find('div', {'data-target': 'DailyPapers'})
        if hydrater_div and isinstance(hydrater_div, Tag) and 'data-props' in hydrater_div.attrs:
            props_string = hydrater_div['data-props']
            # Decode HTML entities like "
            # from html import unescape # Not needed if json.loads handles it
            # props_string = unescape(props_string)

            logger.info("Found data-props attribute for DailyPapers.")
            # logger.debug(f"Props string (first 500 chars): {props_string[:500]}") # Debug log

            data = json.loads(props_string)

            if 'dailyPapers' in data and isinstance(data['dailyPapers'], list):
                logger.info(f"Found {len(data['dailyPapers'])} papers in JSON data.")
                for paper_entry in data['dailyPapers']:
                    if isinstance(paper_entry, dict) and 'paper' in paper_entry and isinstance(paper_entry['paper'], dict):
                        paper_info = paper_entry['paper']
                        arxiv_id = paper_info.get('id')
                        upvotes = paper_info.get('upvotes')
                        # Store other useful info if needed later (like authors list)
                        authors_json = paper_info.get('authors', [])
                        author_names = [a.get('name') for a in authors_json if isinstance(a, dict) and a.get('name')]

                        if arxiv_id:
                            paper_data_from_json[arxiv_id] = {
                                'upvotes': upvotes,
                                'authors': author_names # Store structured authors if available
                                # Add more fields from JSON if desired
                            }
                        else:
                             logger.warning("Found paper entry in JSON without an 'id'.")
            else:
                 logger.warning("'dailyPapers' key not found or not a list in parsed JSON data.")

        else:
            logger.error("Could not find the SVELTE_HYDRATER div with data-target='DailyPapers' or its data-props.")

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from data-props: {e}")
        # logger.debug(f"Problematic props string (first 500 chars): {props_string[:500]}") # Debug log
    except Exception as e:
        logger.error(f"An unexpected error occurred during JSON extraction: {e}", exc_info=True)

    if not paper_data_from_json:
        logger.warning("No paper data extracted from JSON. Upvotes will likely be null.")

    # --- Process Visual Article Tags (Mainly for Links and fallback Authors) ---
    papers_output: List[Dict] = []
    seen_ids = set()
    temp_pdf_dir = "temp_pdfs"
    os.makedirs(temp_pdf_dir, exist_ok=True)

    article_tags = soup.find_all("article", class_="border")
    if not article_tags:
        logger.warning("No article tags found with class 'border'. No papers will be processed.")
        return

    logger.info(f"Found {len(article_tags)} potential paper article tags for processing links/PDFs.")

    for article in article_tags:
        # Find Title and Arxiv ID from the visual elements
        title_tag = article.find("a", class_="line-clamp-3")
        if not title_tag or not isinstance(title_tag, Tag):
            continue

        title = title_tag.text.strip()
        link = title_tag.get("href", "")
        arxiv_id_match = re.search(r"/papers/(\d+\.\d+)", link)
        if not arxiv_id_match:
            continue
        arxiv_id = arxiv_id_match.group(1)

        # Check Duplicates
        if arxiv_id in seen_ids:
            continue
        seen_ids.add(arxiv_id)
        logger.info(f"Processing paper: ID={arxiv_id}, Title='{title[:50]}...'")

        # --- Get Upvotes and Authors from the pre-parsed JSON data ---
        json_data_for_paper = paper_data_from_json.get(arxiv_id)
        upvotes: Optional[int] = None
        authors_str: str = "Not Found"

        if json_data_for_paper:
            upvotes = json_data_for_paper.get('upvotes')
            authors_list = json_data_for_paper.get('authors')
            if authors_list:
                authors_str = ", ".join(authors_list)
            logger.info(f"  -> Data from JSON: Upvotes={upvotes}, Authors found={bool(authors_list)}")
        else:
            logger.warning(f"  -> [{arxiv_id}] No corresponding data found in pre-parsed JSON.")
            # Fallback attempt for authors from visual elements (optional)
            # (You can re-add the visual author scraping here if needed as a fallback)


        # --- PDF Download and Append ---
        full_link = f"https://arxiv.org/abs/{arxiv_id}"
        pdf_filename = f"{arxiv_id}.pdf"
        pdf_path = os.path.join(temp_pdf_dir, pdf_filename)

        if download_pdf(arxiv_id, pdf_path):
            paper_output_data = {
                "title": title,
                "arxiv_id": arxiv_id,
                "authors": authors_str, # Use authors from JSON or fallback
                "link": full_link,
                "pdf_path": pdf_path,
                "upvotes": upvotes, # Use upvotes from JSON
            }
            papers_output.append(paper_output_data)
            logger.info(f"  -> [{arxiv_id}] Added paper to output list.")
        else:
            logger.warning(f"  -> [{arxiv_id}] Failed to download PDF. Skipping paper output entry.")

    # --- JSON Writing ---
    date = datetime.now().strftime("%Y-%m-%d")
    data_dir = "data"
    logger.info(f"Ensuring data directory exists: {data_dir}")
    os.makedirs(data_dir, exist_ok=True)
    data_file_path = os.path.join(data_dir, f"{date}_papers.json")

    logger.info(f"Writing data for {len(papers_output)} papers to {data_file_path}")
    try:
        with open(data_file_path, "w", encoding='utf-8') as f:
            json.dump(papers_output, f, indent=2, ensure_ascii=False)
        logger.info(f"Successfully saved papers information.")
        return papers_output
    except IOError as e:
        logger.error(f"Failed to write JSON file {data_file_path}: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while writing JSON: {e}")

if __name__ == "__main__":
    hf = pull_hf_daily()