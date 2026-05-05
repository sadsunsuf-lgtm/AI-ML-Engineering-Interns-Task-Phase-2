Task 2: End-to-End ML Pipeline with Scikit-learn
🎯 Objective
To build a reusable, production-ready machine learning pipeline for predicting customer churn. The goal was to ensure the model is fully exportable, allowing for seamless transition from training to a real-world application environment.

🛠️ Methodology & Approach
Data Cleaning: Preprocessed the Telco Churn dataset by handling missing values in the TotalCharges column and removing non-predictive features like customerID.

Pipeline Construction:

Numerical Data: Scaled features using StandardScaler.

Categorical Data: Transformed text categories using OneHotEncoder.

Automation: Used ColumnTransformer and the Pipeline API to bundle preprocessing and model training into a single workflow.

Model Development:

Compared Logistic Regression and Random Forest models.

Applied GridSearchCV for intensive hyperparameter tuning to find the best model settings.

Serialization: Exported the complete pipeline using joblib for model reusability.

📈 Key Results & Insights
Best Model: Logistic Regression.

Accuracy: 78.7%.

Observations: The model shows that contract type and monthly charges are significant indicators of customer churn.

Visualizations: Included a Confusion Matrix to evaluate precision and recall, ensuring the model's reliability in identifying actual churners.

💡 Skills Gained
Constructing production-grade ML pipelines using Scikit-learn.

Hyperparameter optimization via GridSearch.

Model export and production-readiness practices.

Data cleaning and visualization for binary classification.

📁 Repository Structure
Task2_Customer_Churn_Pipeline.ipynb: The complete development notebook.

churn_pipeline_model.joblib: The exported, ready-to-use model file.

README.md: Project documentation.
