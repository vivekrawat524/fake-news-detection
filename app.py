import streamlit as st
import pickle

# Load saved model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

st.title("Fake News Detection")

user_input = st.text_area("Enter news text to check:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        vect_text = vectorizer.transform([user_input])
        prediction = model.predict(vect_text)[0]
        result = "Real News ✅" if prediction == 1 else "Fake News ❌"
        st.success(f"Prediction: {result}")
