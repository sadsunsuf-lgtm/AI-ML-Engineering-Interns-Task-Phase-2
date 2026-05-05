# Task 2: End-to-End ML Pipeline with Scikit-learn

## 🎯 Objective
To develop a reusable and production-ready machine learning pipeline for predicting customer churn. This task demonstrates how to handle data preprocessing, model training, and hyperparameter tuning within a single, exportable workflow using the Scikit-learn Pipeline API.

## 🛠️ Methodology & Approach
1. **Data Cleaning**: Handled missing values in the `TotalCharges` column and dropped non-predictive features like `customerID`.
2. **Pipeline Construction**:
   - **Numerical Data**: Applied `StandardScaler` for feature scaling.
   - **Categorical Data**: Applied `OneHotEncoder` to transform text-based categories.
   - **Integration**: Used `ColumnTransformer` to combine these steps into a unified preprocessing layer.
3. **Model Development**:
   - Compared **Logistic Regression** and **Random Forest** models.
   - Used **GridSearchCV** to find the optimal hyperparameters for the models.
4. **Export**: Used `joblib` to save the entire pipeline, including the preprocessing logic and the best-performing model weights.

## 📈 Key Results
* **Best Model**: Logistic Regression.
* **Accuracy**: **78.7%**.
* **Evaluation**: Developed a Confusion Matrix and Classification Report to measure precision and recall for churners.

## 💡 Skills Gained
* Construction of complex Scikit-learn Pipelines.
* Hyperparameter optimization using GridSearchCV.
* Model serialization and export for production readiness.
* Data cleaning and feature engineering for tabular datasets.

## 📁 Submission Checklist
- [x] Jupyter Notebook with logical flow and comments.
- [x] Exported pipeline file (`churn_pipeline_model.joblib`).
- [x] Visualizations (Confusion Matrix).
- [x] Comprehensive README.md.
