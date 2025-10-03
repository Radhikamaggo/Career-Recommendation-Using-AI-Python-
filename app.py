import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json

# Load data
with open('data/career_profiles.json') as f:
    data = json.load(f)

df = pd.DataFrame(data)

# Combine all textual information into a single string feature
def combine_text_features(row):
    return f"{row['skills']} {row['education']} {row['interests']} {row['experience']}"

df['combined_features'] = df.apply(combine_text_features, axis=1)

# Vectorize the combined features using CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['combined_features'])

# Encode job titles as labels for classification
le = LabelEncoder()
y = le.fit_transform(df['job_title'])

# Model training
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Streamlit UI
st.title("Career Path Recommendation System")

st.markdown("""
Enter your profile information below to get the top 3 career path recommendations.
""")

# User inputs
skills_input = st.text_input("Enter your skills (comma separated):", "")
education_input = st.selectbox("Select your highest education level:",
                               ['highschool', 'associate', 'bachelor', 'master', 'phd'])
interests_input = st.text_input("Enter your interests (comma separated):", "")
experience_input = st.selectbox("Select your experience level:", ['entry', 'junior', 'mid', 'senior'])

if st.button("Recommend Careers"):
    user_features = f"{skills_input} {education_input} {interests_input} {experience_input}"
    user_vector = vectorizer.transform([user_features])
    pred_probs = model.predict_proba(user_vector)[0]

    def text_to_vector(text):
        skill_vectorizer = CountVectorizer(vocabulary=vectorizer.vocabulary_)
        return skill_vectorizer.transform([text])

    user_skills_vec = text_to_vector(skills_input.lower())
    descriptions = df['skills'].str.lower().values
    career_skills_vec = vectorizer.transform(descriptions)
    similarities = cosine_similarity(user_skills_vec, career_skills_vec)[0]

    combined_scores = 0.7 * pred_probs + 0.3 * similarities
    top3_idx = combined_scores.argsort()[::-1][:3]

    st.subheader("Top 3 Career Recommendations")
    for idx in top3_idx:
        st.markdown(f"### {df.iloc[idx]['job_title']}")
        st.write(df.iloc[idx]['description'])
        st.markdown(f"**Required Skills:** {df.iloc[idx]['skills']}")
        st.markdown(f"**Typical Education:** {df.iloc[idx]['education'].capitalize()}")
        st.markdown(f"**Interests:** {df.iloc[idx]['interests']}")
        st.markdown(f"**Experience Level:** {df.iloc[idx]['experience'].capitalize()}")
        st.markdown("---")

st.sidebar.title("How it works")
st.sidebar.info("""
This system uses a small sample dataset of career profiles with associated skills, education, interests, and experience.

- Text features from these categories are combined and vectorized.
- A Random Forest machine learning model is trained to classify career paths based on these features.
- When given your profile, the model predicts career fit probabilities.
- Recommendations are enhanced using a simple keyword similarity (cosine similarity) between your skills and job-required skills.
- The top 3 highest scored careers are displayed with descriptions for you to explore.
""")