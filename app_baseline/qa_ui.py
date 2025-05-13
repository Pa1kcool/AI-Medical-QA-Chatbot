import streamlit as st
import requests

st.title("🧠 Medical QA Chatbot (Baseline)")

question = st.text_input("Enter a medical question:")

if st.button("Ask"):
    if question.strip():
        with st.spinner("Thinking..."):
            # IMPORTANT: Replace with your EC2's IP (e.g. http://3.90.12.34:8000)
            res = requests.post("http://18.118.247.179:8000/qa", json={"question": question})
            st.write("**Answer:**", res.json()["answer"])
