🎫 Task 5: Auto Tagging Support Tickets using LLM
Objective
Automate the classification of incoming customer support tickets into relevant categories (Billing, Technical, etc.) and provide the Top 3 most probable tags to improve routing efficiency.

Methodology
Synthetic Data Generation: Created a dataset of 10 diverse support tickets covering various business domains.

LLM Engine: Used Llama-3.1-8b via the Groq API.

Prompt Engineering:

Zero-Shot: Tested the model's baseline ability to categorize text.

Few-Shot: Provided context-specific examples to guide the model's tone and formatting.

Evaluation: Compared the accuracy of the primary tag against the "Actual Category" and visualized the performance boost gained from Few-Shot learning.

Key Observations
Few-Shot learning reduced "hallucinations" in formatting.

The model effectively ranked overlapping issues (e.g., a "Technical" issue that also involves "Billing").
