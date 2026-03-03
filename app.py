import streamlit as st
import joblib
import numpy as np
import time

# ---------------------------------------------------
# Page Config
# ---------------------------------------------------
st.set_page_config(page_title="IPC Legal Assistant", page_icon="⚖️")

# ---------------------------------------------------
# Load Model
# ---------------------------------------------------
model = joblib.load("ipc_model.pkl")
encoder = joblib.load("label_encoder.pkl")

calibrated = model.calibrated_classifiers_[0]
pipe = calibrated.estimator
vectorizer = pipe.named_steps["tfidf"]
svm_model = pipe.named_steps["svm"]

# ---------------------------------------------------
# Sidebar
# ---------------------------------------------------
st.sidebar.title("⚖️ IPC Assistant")
menu = st.sidebar.radio("Navigation", ["Chat", "About Model"])

if menu == "About Model":
    st.title("📊 Model Architecture & Technical Overview")

    st.markdown("## 🧠 Problem Statement")
    st.write("""
    This system performs multi-class text classification to predict the applicable 
    Indian Penal Code (IPC) section based on a given crime description. 
    The goal is to build an explainable AI system capable of assisting in 
    preliminary legal categorization.
    """)

    st.markdown("## ⚙️ Model Pipeline")
    st.write("""
    The model follows a structured Natural Language Processing pipeline:

    1. **Text Preprocessing**
       - Lowercasing
       - Noise removal
       - Whitespace normalization

    2. **Feature Engineering**
       - TF-IDF Vectorization
       - Unigrams + Bigrams
       - Stopword removal
       - Feature limiting for noise control

    3. **Classification Model**
       - Linear Support Vector Machine (LinearSVC)
       - Class weight balancing
       - Regularization applied for generalization

    4. **Probability Calibration**
       - CalibratedClassifierCV (Sigmoid method)
       - Enables confidence scoring
    """)

    st.markdown("## 📈 Model Performance")
    st.write("""
    - Cross-Validation Accuracy: ~81%
    - Test Accuracy: ~76%
    - Balanced dataset across 4 IPC sections
    - Confusion matrix evaluated for misclassification patterns
    """)

    st.markdown("## 🔍 Explainability Features")
    st.write("""
    This system provides interpretability through:

    - Top contributing words (feature impact analysis)
    - Confidence percentage scoring
    - Legal-style formal reasoning paragraph
    - Transparent rule-aligned keyword indicators

    This ensures the model is not a black-box system.
    """)

    st.markdown("## ⚖️ IPC Sections Covered")
    st.write("""
    - IPC 302 — Murder  
    - IPC 379 — Theft  
    - IPC 420 — Cheating & Fraud  
    - IPC 506 — Criminal Intimidation  
    """)

    st.markdown("## 🚀 Deployment")
    st.write("""
    - Built using Streamlit
    - Deployed as an interactive legal AI assistant
    - Designed for demonstration, research, and academic purposes
    """)

    st.markdown("## ⚠️ Disclaimer")
    st.warning("""
    This system is for academic and research purposes only.
    It does not replace professional legal advice or judicial interpretation.
    """)

    st.stop()
# ---------------------------------------------------
# Title
# ---------------------------------------------------
st.title("💬IPC Legal Assistant")

# ---------------------------------------------------
# Chat Memory
# ---------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------------------------------
# User Input
# ---------------------------------------------------
user_input = st.chat_input("Enter crime description...")

if user_input:

    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):

        with st.spinner("Analyzing legal elements..."):
            time.sleep(1.2)

        processed = user_input.lower()
        pred_idx = model.predict([processed])[0]
        predicted_ipc = encoder.inverse_transform([pred_idx])[0]
        probs = model.predict_proba([processed])[0]
        confidence = max(probs) * 100

        # ---------------- Feature Contribution ----------------
        text_features = vectorizer.transform([processed])
        coef = svm_model.coef_[pred_idx]
        contributions = text_features.toarray()[0] * coef
        feature_names = vectorizer.get_feature_names_out()

        top_indices = np.argsort(contributions)[-5:][::-1]
        top_words = [
            feature_names[i]
            for i in top_indices if contributions[i] > 0
        ]

        # ---------------- Legal-Style Explanation Generator ----------------
        def generate_legal_explanation(ipc, text, keywords, confidence):

            base_explanations = {
                "IPC 302": "Section 302 of the Indian Penal Code pertains to the offence of murder.",
                "IPC 379": "Section 379 of the Indian Penal Code deals with the offence of theft.",
                "IPC 420": "Section 420 of the Indian Penal Code relates to cheating and dishonestly inducing delivery of property.",
                "IPC 506": "Section 506 of the Indian Penal Code concerns criminal intimidation."
            }

            intro = base_explanations.get(ipc, "The predicted section under IPC applies in this matter.")

            keyword_part = ""
            if keywords:
                keyword_part = (
                    " The classification is supported by the presence of legally significant terms such as "
                    + ", ".join(keywords[:3])
                    + ", which are commonly associated with this offence."
                )

            confidence_part = (
                f" The model expresses a confidence level of approximately {round(confidence,2)}%, "
                "indicating a reasonably strong correlation between the described facts and the statutory elements of this offence."
            )

            closing = (
                " Based on the available description and linguistic indicators, "
                "the facts prima facie align with the essential ingredients required under this section of law."
            )

            return intro + keyword_part + confidence_part + closing


        legal_paragraph = generate_legal_explanation(
            predicted_ipc,
            user_input,
            top_words,
            confidence
        )

        # ---------------- Display Response ----------------
        response_text = f"""
        🎯 **Predicted IPC Section:** {predicted_ipc}  
        📈 **Confidence:** {round(confidence,2)}%
        """

        st.markdown(response_text)

        st.markdown("### 📖 Legal Analysis")
        st.write(legal_paragraph)

        st.markdown("### 🔍 Key Influential Terms")
        if top_words:
            for word in top_words:
                st.write(f"• {word}")
        else:
            st.write("No significant influencing terms detected.")

        # Save assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": response_text + "\n\n" + legal_paragraph
        })

# ---------------------------------------------------
# Footer
# ---------------------------------------------------
st.markdown("---")
st.caption("Built by Samiksha Gaikwad | Explainable NLP Legal Recommendation Project")