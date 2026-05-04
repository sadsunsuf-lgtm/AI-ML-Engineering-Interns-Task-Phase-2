import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. Load from LOCAL files (the ones you uploaded to the Space)
# "." tells the code to look in the same folder as app.py
model_path = "." 

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Map numeric labels to names
id2label = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

# 2. Prediction Logic
def classify_news(headline):
    # We don't use .to(device) here because Hugging Face Free Spaces 
    # run on CPU by default. It's safer and fast enough for inference.
    inputs = tokenizer(headline, return_tensors="pt", truncation=True, padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=-1).item()
    
    return id2label[prediction]

# 3. Attractive UI Design (Gradio Blocks)
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "sans-serif"],
)

with gr.Blocks(theme=theme) as demo:
    gr.Markdown(
        """
        # 📰 BERT News Intelligence
        ### Automated Topic Classification for Modern Newsrooms
        *Analyze headlines instantly into World, Sports, Business, or Sci/Tech.*
        """
    )
    
    with gr.Row():
        with gr.Column(scale=2):
            input_text = gr.Textbox(
                label="News Headline", 
                placeholder="Ex: Tech giant announces new quantum computer...",
                lines=3
            )
            submit_btn = gr.Button("Classify Headline", variant="primary")
            
        with gr.Column(scale=1):
            output_label = gr.Label(label="Predicted Category")

    gr.Examples(
        examples=[
            ["NASA's James Webb Telescope captures stunning new images of deep space."],
            ["The national team secures a 2-1 victory in the final minutes."],
            ["Central banks consider interest rate hikes to combat inflation."]
        ],
        inputs=input_text
    )

    gr.Markdown("---")
    gr.Markdown("Finalized Deployment for AI Engineering Internship - Phase 2.")

    submit_btn.click(fn=classify_news, inputs=input_text, outputs=output_label)

# 4. Launch
demo.launch()
