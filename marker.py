import subprocess
import sys
import json

# 1. Install compatible versions of torch and torchvision (CPU-only for Colab stability)
print("Installing torch and torchvision...")
subprocess.run([
    sys.executable, "-m", "pip", "install", "--upgrade", "--quiet",
    "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cpu"
], check=True)
print("Installation complete.")

# 2. Reinstall marker-pdf to ensure it uses the updated torch stack
print("Reinstalling marker-pdf...")
subprocess.run([
    sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-cache-dir", "marker-pdf"
], check=True)
print("Reinstallation complete.")

# 3. Test the installation
print("Testing marker-single installation...")
try:
    subprocess.run(['marker_single', '--help'], check=True)
    print("marker_single command is available.")
except subprocess.CalledProcessError as e:  
    print(f"Error testing marker_single: {e}")
except FileNotFoundError:
    print("Error: marker_single command not found. Make sure it's in your PATH.")

from datetime import datetime

date = datetime.now().strftime("%Y-%m-%d")

with open(f"data/{date}_papers.json", "r", encoding="utf-8") as f:
        papers = json.load(f)

paper = papers[0]


import subprocess
result = subprocess.run(
    ['marker_single', paper['pdf_path'], '--output_dir', '/content'],
    capture_output=True, text=True
)
print(result.stderr)

subprocess.run([
    sys.executable, "-m", "pip", "install", "--upgrade", "--quiet",
    "langchain_groq", "Pillow", "langchain_core"
], check=True)

import os
import base64
from PIL import Image
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

# Function to encode images
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Function to resize and compress images
def process_image(image_path, max_size=(800, 800)):
    with Image.open(image_path) as img:
        img.thumbnail(max_size)
        base, ext = os.path.splitext(image_path)
        compressed_image_path = f"{base}_compressed{ext}"
        img.save(compressed_image_path, "JPEG", quality=85)
    return compressed_image_path

# Get image paths
image_directory = paper['pdf_path'].split("/")[1].split(".pdf")[0]
# image_directory = "/content/extracted_images"
# image_directory = "/content/figures"

image_paths = [f for f in os.listdir(image_directory) if f.endswith(('.png', '.jpg', '.jpeg'))][:20]

# Initialize Groq model
model = ChatGroq(model_name="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.7, api_key=groq_api)
# model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.7, google_api_key=GEMINI_API_KEY)
# Process each image individually
img_summary = {}
for image_file in image_paths:
    processed_image_path = os.path.join(image_directory, image_file)

    try:
        # Process the image to resize and compress
        # processed_image_path = process_image(image_path)
        base64_image = encode_image(processed_image_path)

        message_content = [
            {"type": "text", "text": f"Describe this technical image in one line only, that description will be used to categorise the image based on a research paper summary {image_file}."},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
        ]

        message = HumanMessage(content=message_content)
        response = model.invoke([message])

        print(f"Image: {processed_image_path}")
        print(response.content)
        img_summary[processed_image_path] = response.content
        print("-" * 50)

    except Exception as e:
        print(f"Error processing {processed_image_path}: {e}")



import google.generativeai as genai
GEMINI_API_KEY = "AIzaSyByWODhjzPWvS1ER8Jd_6MBQ6EsIMUWh4A"
genai.configure(api_key=GEMINI_API_KEY)
import logging
import json
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# with open('config.json', 'r') as f:
#     sent_papers = json.load(f)

# if sent_papers[0]['title'] in hf[0]['title']:
# else:
#   paper = hf[1]
def assign_image(rp_summary: str, img_summary: dict) -> dict:
    """
    Categorizes an image using the Gemini API.

    Args:
      summary: The summary of the image.

    Returns:
      The category of the image.
    """
    prompt = f"""Given the following summary of an image from a research paper and also given a dictionary with image path as key and their image summary as value,
    now analyze the image summary and research papers summary and categorize which image should be used to display with research paper summary.
    If no image is suitable, return "No image".
    Image summary: {rp_summary}
    Image dictionary: {img_summary}
    Output:
    {{
      "image_path": "img2.png",
      "reason": "Diagram of the architecture directly supports the paper's topic."
    }}
    Output should be like this, Just the json object and strictly no other text or any expalanation.
    """
    print("generating response")
    llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.7, api_key="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")
    response = llm.invoke(prompt)
    # Remove markdown json formatting if response is in markdown code block format
    print(" response generated")

    if response.content.startswith("```json"):
        print(" formatting response ")
        logger.warning("Response contains markdown json code format - removing formatting")
        response.content = response.text.replace("```json", "").replace("```", "").strip()
        response.content = json.loads(response.content)
    print(response.content)
    return response.content


def summarize_paper(title: str, authors: str, pdf_path: str, model_name: str) -> tuple[str, str]:
    """
    Summarizes a research paper and determines its category using the Gemini API.

    Args:
    - title (str): The title of the paper.
    - authors (str): The authors of the paper.
    - pdf_path (str): The path to the PDF of the paper.

    Returns:
    - tuple[str, str]: The summary and category of the paper.
    """

    model = genai.GenerativeModel(model_name=model_name)

    # Upload the PDF to Gemini
    pdf_file = genai.upload_file(path=pdf_path, display_name=f"paper_{title}")

    # Load the prompt template
    with open("prompt.md", "r") as f:
        prompt_template = f.read()

    prompt = prompt_template.replace("{title}", title).replace("{authors}", authors)

    response = model.generate_content([pdf_file, prompt])
    response_text = response.text

    logger.info(f"Response received: {response_text}")

    # Remove markdown json formatting if response is in markdown code block format
    if response_text.startswith("```json"):
        logger.warning("Response contains markdown json code format - removing formatting")
        response_text = response_text.replace("```json", "").replace("```", "").strip()

    response_data = json.loads(response_text)

    return response_data["summary"], response_data["category"]

summaries={}
try:
    # time.sleep(60) # Sleep for 1 minute to avoid rate limiting

  summary, category = summarize_paper(
      title=paper["title"],
      authors=paper["authors"],
      pdf_path=paper["pdf_path"],
      model_name= "gemini-2.5-flash-preview-05-20"# Not free "gemini-2.5-pro-preview-05-06"           #"gemini-2.0-pro-exp-02-05"
  )

  summaries.update({**paper, "summary": summary, "category": category, **json.loads(assign_image(summary, img_summary))})
except Exception as e:
    print(e)
    try:
        logger.warning(f"Failed to summarize paper {paper['title']}. Trying with a different model.")
        summary, category = summarize_paper(
            title=paper["title"],
            authors=paper["authors"],
            pdf_path=paper["pdf_path"],
            model_name="gemini-2.0-flash-001"
        )
        summaries.update({**paper, "summary": summary, "category": category, **json.loads(assign_image(summary, img_summary))})
    except Exception as e:
        logger.error(f"Failed to summarize paper {paper['title']} with both models. Due to {e}")


title = summaries['title']
# image_path = "/content/2505.00174/_page_12_Figure_0.jpeg"
image_path = summaries['image_path']
summary = summaries['summary']
authors = summaries['authors']
pdf_url = summaries['link']


import smtplib
import datetime
import markdown
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

# Sender and receiver details
from_address = "harsh.mishra@analyticsvidhya.com"
password = "wtfh wdqd jglv wfcn"  # Use App Password

to_addresses = ["harsh.mishra@analyticsvidhya.com"]

# Get today's date in desired format
today_date = datetime.datetime.now().strftime("%d %b %Y")  # Example: "20 Dec 2024"

# Generate dynamic subject
email_subject = f"{today_date}'s latest research papers"

# Convert Title and Summary from Markdown to HTML
title_html = markdown.markdown(title)
summary_html = markdown.markdown(summary, extensions=["extra", "md_in_html"])
authors_html = markdown.markdown(f"**By:** {authors}")
pdf_link_html = f"""<p> Read Full Paper Here: {pdf_url}</p>'"""

# Define HTML email template
html_content = f"""
<html>
  <body>
    <h2>{title_html}</h2>
    <p>{authors_html}</p>
    <img src="cid:image1" alt="Embedded Image" width="500"><br><br>
    {summary_html}<br>
    {pdf_link_html}<br><br>
<hr>
  </body>
</html>
"""

# Creating the email
msg = MIMEMultipart()
msg['From'] = from_address
msg['To'] = ", ".join(to_addresses)
msg['Subject'] = email_subject  # Set dynamic subject

# Attach HTML content
msg.attach(MIMEText(html_content, 'html'))

# Attach the image inline
try:
  with open(image_path, "rb") as img_file:
      img = MIMEImage(img_file.read())
      img.add_header("Content-ID", "<image1>")  # Matches the cid in HTML
      img.add_header("Content-Disposition", "inline", filename="image.jpeg")
      msg.attach(img)
except:
  print("No Image found")
finally:
  pass

# Send email via SMTP
s = smtplib.SMTP('smtp.gmail.com', 587)
s.starttls()
s.login(from_address, password)
s.sendmail(from_address, to_addresses, msg.as_string())
s.quit()

print("Email sent successfully!")
