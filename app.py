import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(page_title="Quick Personality Assessment", layout="centered")

# Title and Sidebar
st.sidebar.title("🧭 Navigation")
page = st.sidebar.selectbox("Go to", ["🔍 Prediction", "📊 Graphs", "📄 Raw Dataset"])

# Cache the model load
@st.cache_resource
def load_model():
    return joblib.load("personality_model.pkl")

# Cache the data load
@st.cache_data
def load_data():
    df = pd.read_csv("data-final.csv", sep='\t', low_memory=False)
    question_cols = [col for col in df.columns if col.startswith(("EXT", "EST", "AGR", "CSN", "OPN")) and df[col].dtype != "object"]
    df = df[question_cols].dropna()
    df["Extraversion"] = df[[f"EXT{i}" for i in range(1, 11)]].mean(axis=1)
    df["Neuroticism"] = df[[f"EST{i}" for i in range(1, 11)]].mean(axis=1)
    df["Agreeableness"] = df[[f"AGR{i}" for i in range(1, 11)]].mean(axis=1)
    df["Conscientiousness"] = df[[f"CSN{i}" for i in range(1, 11)]].mean(axis=1)
    df["Openness"] = df[[f"OPN{i}" for i in range(1, 11)]].mean(axis=1)
    return df

model = load_model()
df = load_data()

# Trait labels
trait_names = {
    "Openness": "Creativity & Curiosity",
    "Conscientiousness": "Self-Discipline & Responsibility",
    "Extraversion": "Sociability & Energy",
    "Agreeableness": "Kindness & Cooperation",
    "Neuroticism": "Emotional Sensitivity"
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
    answers = []
    for idx, q in enumerate(questions):
        ans = st.radio(q, options=list(answer_mapping.keys()), key=f"q{idx}")
        answers.append(answer_mapping[ans])

    def predict_personality(answers):
        input_array = np.array(answers).reshape(1, -1)
        return model.predict(input_array)

    def generate_summary(prediction):
        summary = ""
        traits = prediction[0]
        if traits[0] > 3:
            summary += "You are highly curious and imaginative. "
        elif traits[0] > 2:
            summary += "You are moderately curious and open-minded. "
        else:
            summary += "You prefer familiar experiences over new ones. "

        if traits[1] > 3:
            summary += "You're organized and dependable. "
        elif traits[1] > 2:
            summary += "You are fairly responsible and structured. "
        else:
            summary += "You tend to go with the flow and dislike rigid schedules. "

        if traits[2] > 3:
            summary += "You enjoy being around others and thrive in social settings. "
        elif traits[2] > 2:
            summary += "You enjoy a balance of social and quiet time. "
        else:
            summary += "You are reserved and value alone time. "

        if traits[3] > 3:
            summary += "You are empathetic and cooperative. "
        elif traits[3] > 2:
            summary += "You are generally kind and fair. "
        else:
            summary += "You may be more direct and goal-focused. "

        if traits[4] > 3:
            summary += "You may feel stress or worry more frequently. "
        elif traits[4] > 2:
            summary += "You manage emotions fairly well. "
        else:
            summary += "You are emotionally stable and calm under pressure. "

        return summary

    if st.button("🔍 Predict Personality"):
        with st.spinner("Analyzing your responses..."):
            prediction = predict_personality(answers)
            st.subheader("🧾 Trait Scores")
            for i, trait in enumerate(trait_names.keys()):
                st.markdown(f"- **{trait_names[trait]}**: `{round(prediction[0][i], 2)} / 5`")

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
    st.dataframe(df.head(100))  # Show only top 100 records
