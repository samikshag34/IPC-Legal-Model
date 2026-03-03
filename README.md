# ⚖️ AI-Based IPC Legal Assistant

An Explainable NLP-based Machine Learning system that predicts the applicable 
Indian Penal Code (IPC) section from a given crime description.

🔗 **Live Demo:** https://your-streamlit-link.streamlit.app

---

## 🧠 Problem Statement

Legal case descriptions often require preliminary categorization under the appropriate IPC section.  
This project builds a multi-class text classification model to assist in identifying the most relevant IPC section based on textual input.

The system also provides explainability through feature contribution analysis and formal legal-style reasoning.

---

## 🚀 Features

- ✅ Multi-class IPC section prediction
- ✅ TF-IDF + Linear SVM classifier
- ✅ Calibrated probability confidence scoring
- ✅ Legal-style explanation paragraph generation
- ✅ Top contributing word analysis (Explainable AI)
- ✅ Chat-style interactive UI
- ✅ Downloadable PDF case report
- ✅ Streamlit Cloud deployment

---

## ⚙️ Model Architecture

### 1️⃣ Text Preprocessing
- Lowercasing
- Noise removal
- Stopword removal
- Token normalization

### 2️⃣ Feature Engineering
- TF-IDF Vectorization
- Unigrams + Bigrams
- Limited feature space to reduce noise

### 3️⃣ Classification Model
- Linear Support Vector Machine (LinearSVC)
- Class-balanced training
- Regularization for generalization

### 4️⃣ Probability Calibration
- CalibratedClassifierCV (Sigmoid method)
- Enables confidence scoring

---

## 📊 Model Performance

- Cross-Validation Accuracy: ~81%
- Test Accuracy: ~76%
- Balanced dataset across 4 IPC sections
- Evaluated using confusion matrix and classification report

---

## ⚖️ IPC Sections Covered

- IPC 302 — Murder
- IPC 379 — Theft
- IPC 420 — Cheating & Fraud
- IPC 506 — Criminal Intimidation

---

## 📁 Repository Structure
ipc-legal-ai/
│
├── app.py → Streamlit deployment app
├── notebooks/ → Model training & evaluation notebook
├── ipc_model.pkl → Trained ML model
├── label_encoder.pkl → Label encoder
├── requirements.txt → Dependencies
└── README.md


## 🌍 Deployment

The application is deployed using:

- Streamlit
- GitHub
- Streamlit Cloud

The deployment automatically rebuilds when new commits are pushed.

---

## 🔍 Explainability

This system is designed to avoid black-box behavior by:

- Displaying top contributing words
- Generating legal-style reasoning
- Providing confidence percentage
- Transparent classification logic

---

## ⚠️ Disclaimer

This system is developed for academic and research purposes only.  
It does not constitute legal advice and should not replace professional legal consultation.

---

## 👩‍💻 Author

**Samiksha Gaikwad**  
B.Sc Data Science  
NLP | Machine Learning | Explainable AI