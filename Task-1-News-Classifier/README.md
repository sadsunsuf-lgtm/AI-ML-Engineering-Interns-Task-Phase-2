Task 1: News Topic Classifier Using BERT
🚀 Live Demo
Interact with the live model here: https://huggingface.co/spaces/SanaNasir/News-Topic-Classifier

🎯 Objective
The goal of this task was to develop a high-performance Natural Language Processing (NLP) system capable of automatically categorizing news headlines into four distinct topics: World, Sports, Business, and Sci/Tech. This involved leveraging Transfer Learning by fine-tuning a pre-trained BERT model to handle real-world text classification.

📊 Dataset
The AG News Dataset (via Hugging Face Datasets) was used for this project. It consists of thousands of news articles categorized into four classes, providing a robust foundation for training and evaluation.

🛠️ Methodology & Approach
Tokenization & Preprocessing:

Utilized AutoTokenizer from bert-base-uncased.

Applied padding and truncation to handle varying headline lengths.

Model Development:

Fine-tuned the bert-base-uncased transformer model.

Implemented a sequence classification head for the 4 target labels.

Training & Evaluation:

Trained using the Hugging Face Trainer API in Google Colab.

Evaluated performance using Accuracy and F1-Score to ensure balanced classification.

Deployment:

Designed an attractive UI using Gradio Blocks.

Deployed a permanent, "lifetime free" version on Hugging Face Spaces.

📈 Key Results & Observations
Performance: The model achieved high accuracy, effectively distinguishing between technical and sporting news.

Insight: Fine-tuning BERT proved significantly more effective than traditional ML methods, as the model understands the context of words rather than just frequency.

Deployment: Moving from Colab to Hugging Face required careful handling of model serialization (saving model.safetensors and tokenizer.json) and environment dependencies.

💡 Skills Gained
NLP & Transformers: Deep dive into the BERT architecture.

Transfer Learning: Adapting large-scale pre-trained models to specific tasks.

End-to-End Deployment: Managing the full lifecycle from training to a live URL.

📁 Repository Structure
Task1_News_Topic_Classifier_BERT.ipynb: Training and evaluation notebook.

app.py: Gradio interface code for deployment.

requirements.txt: Environment dependencies for Hugging Face.
