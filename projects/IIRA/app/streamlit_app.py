import streamlit as st
import requests

st.title("🔍 Intelligent Research Agent Interface")

# Input from user
user_query = st.chat_input("Ask me anything...")

if user_query:
    with st.chat_message("user"):
        st.write(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Calling API..."):
            response = requests.post(
                "http://localhost:8000/research", 
                params={"user_query": user_query}
            )
            
            if response.status_code == 200:
                report = response.json().get("report")
                st.markdown(report)
            else:
                st.error("Failed to connect to the Research API.")