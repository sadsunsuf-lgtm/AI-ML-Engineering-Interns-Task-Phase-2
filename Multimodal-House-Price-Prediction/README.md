# Multimodal AI: Housing Price Prediction (Images + Tabular)

## 🎯 Objective
This project aims to predict real estate market values by combining two different types of data (modalities):
1. **Structured Data:** Numerical stats like bedrooms, bathrooms, and square footage.
2. **Visual Data:** Frontal images of houses to capture "curb appeal" and architectural features.

This approach mimics how human experts evaluate properties—by looking at both the numbers and the actual appearance.

## 🛠 Methodology & Approach
- **Image Processing:** Used a Convolutional Neural Network (CNN) to extract visual features from 64x64 property images.
- **Structured Processing:** Scaled numerical features and processed them through a Feed-Forward Neural Network.
- **Feature Fusion:** Combined the outputs of both branches into a single "Concat" layer.
- **Regression Head:** A final dense layer outputs the predicted price in USD.

## 📊 Key Results & Observations
- **Architecture:** Successfully implemented a multi-input Keras model.
- **Insights:** The model learned that image features provide a significant "vibe" weight to the price that numerical data alone cannot capture.
- **Metrics:** Evaluated using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) to ensure prediction accuracy.

