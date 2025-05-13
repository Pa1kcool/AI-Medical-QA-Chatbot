import streamlit as st
import requests

st.set_page_config(page_title="🧠 MedQA - MLOps", layout="centered")
st.title("🩺 Medical QA Chatbot (MLOps Version)")

st.markdown("Ask any medical question based on **MedQuAD** fine-tuned model. All requests are logged via ClearML.")

question = st.text_input("Enter a medical question:")

if st.button("Ask"):
    if question.strip():
        with st.spinner("Querying the LLM..."):
            try:
                # Replace with your EC2 public IP or domain
                backend_url = "http://18.191.204.215:8001/qa"
                response = requests.post(backend_url, json={"question": question})
                response.raise_for_status()
                answer = response.json()["answer"]
                st.success("✅ Answer:")
                st.write(answer)
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
