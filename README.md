# Sentiment Analysis of Bangla Social Media Comments: BERT vs Random Forest

## 📌 Introduction
This project focuses on classifying Bangla social media content using two different approaches and comparing their effectiveness:
- **Bangla-BERT**: A Transformer-based model leveraging contextual embeddings.
- **Random Forest**: A traditional machine learning algorithm for fast prototyping.

---

## 🤖 Why Two Approaches?

| Approach | Why Use It |
|---------|-------------|
| **Bangla-BERT** | Handles complex and messy text using context-aware embeddings. Ideal for high-accuracy, production-grade systems. |
| **Random Forest** | Fast, easy to train, and interpretable. Useful for quick iterations and lightweight environments. |

---

## 🔧 Training Pipeline Comparison

### 🟩 Random Forest Pipeline:
1. Data Loading & Splitting  
2. Intensive Text Preprocessing  
3. Class Balancing with `RandomOverSampler`  
4. TF-IDF Vectorization (Unigrams + Bigrams)  
5. Hyperparameter Tuning with Optuna  
6. Model Training  
7. Evaluation  

### 🟦 Bangla-BERT Pipeline:
1. Data Loading & Basic Preprocessing  
2. Tokenization & Dataset Wrapping  
3. Dynamic Batching with Imbalance Handling  
4. Model Initialization  
5. Training Utilities  
6. Hyperparameter Tuning with Optuna  
7. Threshold Optimization  
8. Final Training & Evaluation  
9. ONNX Export for Deployment  

---

## 🧠 How Bangla Text Classification is Handled Differently

| Step | Random Forest | Bangla-BERT |
|------|----------------|--------------|
| **Data Prep** | Heavy cleaning + oversampling | Minimal cleaning + smart sampling |
| **Text → Numbers** | TF-IDF (10K sparse features) | 768-dim contextual embeddings |
| **Class Imbalance** | Oversampling + Class Weights | Smart batch sampling + Loss weighting |

---

## ⚖️ Model Choice Guide

Choose **Random Forest** when:
- You need a lightweight model
- You want explainability
- You can afford to spend time cleaning the text

Choose **Bangla-BERT** when:
- You need **high accuracy**
- You have GPU access
- You deal with **noisy real-world** text
- You want **minimal preprocessing**

---

## 📊 Results

| Category     | #Examples | BERT Score | RF Score | Difference |
|--------------|-----------|------------|----------|------------|
| Not Bully    | 15,340    | 85%        | 64%      | +21%       |
| Troll        | 10,462    | 77%        | 49%      | +28%       |
| Sexual       | 8,928     | 84%        | 43%      | +41%       |
| Religious    | 7,577     | 92%        | 48%      | +44%       |
| Threat       | 1,694     | 79%        | 20%      | +59%       |

### 🏁 Overall Accuracy:
- **Bangla-BERT**: 84%  
- **Random Forest**: 51%

---

## ⚠️ Limitations
- Low sample count for rare labels like "threat"
- BERT requires GPU and high memory for training

---

## 🚀 Future Improvements
- Collect more rare-category samples (e.g., "threat")
- Apply **knowledge distillation** to make BERT lighter
- Add explainability features (e.g., attention heatmaps)
- Integrate user feedback for model retraining

---

## 🔌 Running the API

### ▶️ Start the FastAPI Server
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
