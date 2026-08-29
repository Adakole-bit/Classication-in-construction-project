st.title("📝 Text Classification App")

# Load model and vectorizer separately using st.cache_resource
@st.cache_resource
def load_assets():
    model = joblib.load("model.joblib")
    vectorizer = joblib.load("vectorizer.joblib")
    return model, vectorizer


user_text = st.text_area("Enter text to classify:", "")

if st.button("Classify Text"):
    if user_text.strip():
        # Transform the input text to numeric features first
        text_vectorized = vectorizer.transform([user_text])
        
        # Predict class and probabilities
        prediction = model.predict(text_vectorized)
        
        st.subheader("Result")
        st.write(f"**Predicted Category:** {prediction[0]}")
        
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(text_vectorized)
            confidence = max(probabilities[0]) * 100
            st.write(f"**Confidence:** {confidence:.2f}%")
    else:
        st.warning("Please enter some text before submitting.")

