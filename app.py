import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(page_title="Quick Personality Assessment", layout="centered")
st.sidebar.title("🧭 Navigation")
page = st.sidebar.selectbox("Go to", ["🔍 Prediction", "📊 Graphs", "📄 Raw Dataset"])

# Load dataset and model
@st.cache_data
def load_data():
    return pd.read_csv("personality_data.csv")

@st.cache_resource
def load_model():
    return joblib.load("personality_model.pkl")

df = load_data()
model = load_model()

# Trait label mapping
trait_names = {
    "openness": "Creativity & Curiosity",
    "conscientiousness": "Self-Discipline & Responsibility",
    "extraversion": "Sociability & Energy",
    "agreeableness": "Kindness & Cooperation",
    "neuroticism": "Emotional Sensitivity"
}

# Prediction Page
if page == "🔍 Prediction":
    st.title("🧠 Quick Personality Assessment")
    st.markdown("Answer the following 10 questions to receive an estimate of your personality traits.")

    questions = [
        "How comfortable are you with trying new and unfamiliar things?",
        "How often do you feel nervous or anxious?",
        "How much do you enjoy socializing?",
        "How often do you follow a set routine or plan in your daily activities?",
        "How do you approach tasks or goals that require a lot of effort and time?",
        "How open are you to new ideas?",
        "How easy is it for you to understand and share the feelings of others?",
        "How often do you worry about future events?",
        "How much do you enjoy being the center of attention?",
        "How would you rate your ability to make decisions quickly?"
    ]

    answer_mapping = {"Never": 0, "Rarely": 1, "Sometimes": 2, "Often": 3, "Always": 4}
    answers = [answer_mapping[st.radio(q, list(answer_mapping.keys()), key=q)] for q in questions]

    def predict_personality(answers):
        inputs = np.array(answers).reshape(1, -1)
        return model.predict(inputs)

    def generate_summary(pred):
        summary = ""
        traits = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
        texts = [
            ["You are highly curious and imaginative. ", "You are moderately curious and open-minded. ", "You prefer familiar experiences over new ones. "],
            ["You're organized and dependable. ", "You are fairly responsible and structured. ", "You tend to go with the flow and dislike rigid schedules. "],
            ["You enjoy being around others and thrive in social settings. ", "You enjoy a balance of social and quiet time. ", "You are reserved and value alone time. "],
            ["You are empathetic and cooperative. ", "You are generally kind and fair. ", "You may be more direct and goal-focused. "],
            ["You may feel stress or worry more frequently. ", "You manage emotions fairly well. ", "You are emotionally stable and calm under pressure. "]
        ]
        for i in range(5):
            val = pred[0][i]
            if val > 3:
                summary += texts[i][0]
            elif val > 2:
                summary += texts[i][1]
            else:
                summary += texts[i][2]
        return summary

    if st.button("🔍 Predict Personality"):
        prediction = predict_personality(answers)
        st.subheader("🧾 Results:")
        traits = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
        for i, trait in enumerate(traits):
            score = round(prediction[0][i], 2)
            st.markdown(f"- **{trait_names[trait]}**: `{score} / 5`")

        st.subheader("🧠 Personality Summary")
        st.write(generate_summary(prediction))

        st.subheader("🧭 Overall Hint")
        if prediction[0][0] > 2 and prediction[0][2] > 2:
            st.success("You seem to be a curious, energetic, and adventurous person!")
        else:
            st.info("You appear to be more thoughtful and introspective.")

# Graphs Page
elif page == "📊 Graphs":
    st.title("📊 Personality Trait Distributions")
    selected_trait = st.selectbox("Select a trait to view distribution", list(trait_names.keys()))
    fig, ax = plt.subplots()
    sns.histplot(df[selected_trait], bins=30, kde=True, ax=ax, color="skyblue")
    ax.set_title(f"{trait_names[selected_trait]} Score Distribution")
    ax.set_xlabel("Score (0–5)")
    st.pyplot(fig)

# Raw Dataset Page
elif page == "📄 Raw Dataset":
    st.title("📄 View Raw Dataset")
    st.dataframe(df.head(100))
