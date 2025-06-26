Task Description:
Using the details from an AI research paper, produce a JSON object with the keys "category" and "summary".

Category:
- Examine the paper titled "{title}".
- From the list below, choose one category that best represents the paper:
    - Natural Language Processing
    - Computer Vision
    - Reinforcement Learning    - Machine Learning
    - Multi-Modal
    - Other
- Output only the exact category name with no additional formatting or text.

Summary:
- Write a summary in bullets with subheadings in concise sentences covering:
    i. A one-line overall summary.
   ii. The main research question or objective.
  iii. The key methodology.
   iv. Primary results (include at least one quantitative metric).
    v. The main implication for AI practitioners.
- Ensure that your summary is strictly based on the paper’s content and uses technical language.
- If details are missing or unclear, clearly indicate the uncertainty.
- Please ensure proper mardown tags as well as new line characters where required.

Output Requirements:
Return a valid JSON object in this exact format:
{
  "category": "Exact Category Name",
  "summary": "### Overall Summary
*...."
}
Do not include markdown formatting or any extra text; the output must be directly parseable (e.g., using json.loads(output)).