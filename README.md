# Task 1: News Topic Classifier Using BERT

## Objective
To fine-tune a pre-trained BERT (`bert-base-uncased`) model to automatically classify news headlines into four categories: World, Sports, Business, and Sci/Tech using the AG News dataset.

## Methodology
1. **Data Preprocessing:** Tokenized the AG News dataset using the BERT Fast Tokenizer with a max length of 512.
2. **Model Training:** Fine-tuned the model using the Hugging Face `Trainer` API for 2 epochs on a T4 GPU.
3. **Optimization:** Used a learning rate of 2e-5 and weight decay of 0.01 to ensure model stability.
4. **Evaluation:** Validated the model using Accuracy and F1-score metrics.

## Key Results
* **Accuracy:** 91.20%
* **F1-Score:** 91.28%
* **Deployment:** Successfully built a live inference interface using Gradio.

## How to Run
1. Open the `.ipynb` file in Google Colab.
2. Ensure the Hardware Accelerator is set to **T4 GPU**.
3. Run all cells to train the model and launch the Gradio interface.
