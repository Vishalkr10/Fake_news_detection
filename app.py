import streamlit as st
import joblib
import pandas as pd

# Load model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Single News Check", "Bulk News Check"])

st.title("📰 Fake News Detection System")
st.write("Detect whether a news article is real or fake using Machine Learning.")

# Function to predict news
def predict_news(news_text):
    transformed_text = vectorizer.transform([news_text])
    prediction = model.predict(transformed_text)[0]
    probability = model.predict_proba(transformed_text)[0]
    return "Fake News" if prediction == 1 else "Real News", probability

# Single News Check
if page == "Single News Check":
    st.subheader("🔍 Check a Single News Article")
    news_input = st.text_area("Enter the news content below:")
    
    if st.button("Check News"):
        if news_input.strip():
            label, prob = predict_news(news_input)
            st.write(f"### Prediction: **{label}**")
            st.write(f"🟢 Probability (Real News): {prob[0]:.2f}")
            st.write(f"🔴 Probability (Fake News): {prob[1]:.2f}")
        else:
            st.warning("Please enter some text.")

# Bulk News Check
elif page == "Bulk News Check":
    st.subheader("📂 Upload a File for Bulk News Checking")
    uploaded_file = st.file_uploader("Upload a CSV file with a column 'text'", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        if 'text' in df.columns:
            # Apply predictions correctly
            df['Prediction'] = df['text'].apply(lambda x: predict_news(x)[0])
            df['Real Probability'] = df['text'].apply(lambda x: predict_news(x)[1][0])
            df['Fake Probability'] = df['text'].apply(lambda x: predict_news(x)[1][1])

            # Display the first few results
            st.write("### Prediction Summary:")
            st.dataframe(df[['text', 'Prediction', 'Real Probability', 'Fake Probability']])

            # Allow users to download results
            st.download_button("Download Results", df.to_csv(index=False), file_name="predictions.csv", mime="text/csv")

        else:
            st.error("CSV file must contain a 'text' column.")
