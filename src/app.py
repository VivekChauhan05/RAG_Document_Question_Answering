import streamlit as st
from vectorstore import VectorStore
from chatbot import Chatbot
import uuid

def main():
    st.title("Document QA Bot 🤖")
    st.write("Upload a PDF, input your API keys, and ask questions!")

    # Initialize session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Sidebar for API keys
    with st.sidebar:
        st.header("API Keys 🔑")
        cohere_api_key = st.text_input("Cohere API Key", type="password")
        pinecone_api_key = st.text_input("Pinecone API Key", type="password")

    # Upload PDF document
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    user_query = st.text_input("Ask a question based on the document")

    # Process after user submits a query
    if st.button("Submit") and uploaded_file and cohere_api_key and pinecone_api_key:
        with st.spinner("Processing PDF..."):
            with open("uploaded_document.pdf", "wb") as f:
                f.write(uploaded_file.read())

            # Create VectorStore instance
            vectorstore = VectorStore("uploaded_document.pdf", cohere_api_key, pinecone_api_key)

            # Initialize chatbot
            chatbot = Chatbot(vectorstore, cohere_api_key)

            with st.spinner("Generating response..."):
                response, retrieved_docs = chatbot.respond(user_query)

                # Save conversation in session state
                st.session_state["chat_history"].append((user_query, response, retrieved_docs))

    # Display chat history
    if st.session_state["chat_history"]:
        for user_query, response, retrieved_docs in st.session_state["chat_history"]:
            st.write(f"**You:** {user_query}")

            accumulated_response = ""
            for event in response:
                if event.event_type == "text-generation":
                    accumulated_response += event.text
            st.write(f"**Bot:** {accumulated_response}")

if __name__ == "__main__":
    main()
