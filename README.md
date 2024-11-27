# RAG-Based Document Question Answering System

This project implements a **Retrieval-Augmented Generation (RAG) chatbot** that allows users to upload PDF documents, ask questions based on the content, and receive accurate, document-specific answers. It combines the power of **Cohere** for language processing and embeddings, **Pinecone** for efficient vector storage and retrieval, and **Streamlit** for a user-friendly interface.

---

## Features

- **PDF Processing**: Extracts text from uploaded PDF documents and splits it into manageable chunks for embedding and storage.  
- **Embedding and Retrieval**: Uses Cohere's embeddings for encoding document chunks and Pinecone for scalable vector similarity search.  
- **Question Answering**: Leverages Cohere's language models to generate accurate responses by retrieving and analyzing relevant document chunks.  
- **Interactive Interface**: Provides a simple and intuitive interface using Streamlit for uploading documents, entering queries, and viewing results.  

---

## How to Use

Follow the steps below to run and interact with the project:

### 1. Clone the Repository
Clone the repository to your local system using the following command:
```bash
git clone https://github.com/VivekChauhan05/RAG_Document_Question_Answering.git
cd RAG_Document_Question_Answering

### 2. Create and Activate a Virtual Environment
Create a virtual environment and activate it to isolate project dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
 

### 3. Install Dependencies
Install the required Python libraries using the provided requirements.txt file:

```bash
pip install -r requirements.txt

### 4. Obtain API Keys
Get your API keys for:

-Cohere: Sign up at Cohere to obtain an API key.
-Pinecone: Sign up at Pinecone to obtain an API key.
-These keys will be entered via the Streamlit interface when running the app.

### 5. Run the Application

Launch the Streamlit application:

```bash
streamlit run app.py

### 6. Access the Application
Once the application is running, open your browser and navigate to the URL provided by Streamlit, typically http://localhost:8501.

### 7. Upload a Document
Use the interface to upload a PDF file containing the content you want to query.

### 8. Ask Questions
-Enter your question in the query box. The chatbot will:

-Retrieve relevant chunks of text from the uploaded document.
-Generate a precise and context-aware response.

# Project Structure
```bash
.
├── app.py                # Main application file with Streamlit interface
├── vectorstore.py        # Handles PDF processing, embedding, and retrieval
├── chatbot.py            # Handles user interaction and response generation
├── requirements.txt      # Project dependencies
├── README.md             # Project documentation


### Future Enhancements
Add support for multi-language documents.
Enhance the UI with multi-document support and export options for chat history.
Enable deployment to cloud platforms for wider accessibility.
Integrate additional vector databases for broader compatibility.

###License
This project is licensed under the Apache License.

### Acknowledgments
Cohere AI for their powerful embedding and language models.
Pinecone for scalable vector search infrastructure.
Streamlit for making it easy to build interactive data apps.
 
