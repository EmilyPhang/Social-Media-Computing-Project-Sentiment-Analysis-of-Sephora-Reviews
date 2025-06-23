# Sentiment Analysis of Sephora Reviews

This project explores the effectiveness of various machine learning and deep learning models in classifying customer sentiment from Sephora product reviews. The models include Logistic Regression, SVM, Pre-Trained BERT, CNN, and Fine-Tuned BERT. The goal is to evaluate how well each model can detect **positive**, **neutral**, and **negative** sentiments from user-generated text.

---

## Dataset

- **Source**: Sephora product review dataset (https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews)  
- **Classes**: Positive, Neutral, Negative  
- **Split**: 60% Training, 20% Validation, 20% Testing  


---

## Models Used

- **Logistic Regression (TF-IDF)**
- **Support Vector Machine (TF-IDF)**
- **BERT as Feature Extractor (No Fine-Tuning)**
- **Convolutional Neural Network (Trainable Embeddings)**
- **Fine-Tuned BERT (Transformer-based Classification)**

Each model was evaluated using **accuracy**, **precision**, **recall**, and **F1-score**, along with **confusion matrix** analysis.

---

## Results Summary

| Model                  | Accuracy | Precision | Recall | F1-Score |
|-----------------------|----------|-----------|--------|----------|
| Logistic Regression   | 67.69%   | 67.42%    | 67.69% | 67.53%   |
| SVM                   | 65.32%   | 64.92%    | 65.32% | 65.07%   |
| BERT (Feature Extractor) | 69.68% | 70.54% | 69.68% | 69.93%   |
| CNN                   | 71.00%   | 70.00%    | 71.00% | 70.00%   |
| Fine-Tuned BERT       | **79.00%** | **79.00%** | **79.00%** | **79.00%** |

---

## Key Findings

- Fine-Tuned BERT outperformed all other models across all metrics.
- CNN and Pre-Trained BERT also showed strong performance, surpassing traditional models.
- Neutral reviews were generally the most difficult to classify correctly, especially for non-contextual models.

---

## Future Work

Future improvements could include:
- Training on **larger and more diverse datasets** to improve generalization.
- Implementing **hyperparameter tuning** (e.g., grid search, random search) to optimize learning rate, batch size, dropout, and regularization.
- Exploring **transformer variants** like RoBERTa or DistilBERT.
- Using **ensemble methods** for performance boosting.
- Adding **explainability tools** like SHAP or LIME to interpret model decisions, especially in user-facing applications.

---

