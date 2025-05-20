import os
import streamlit as st
from chatbot import load_data, create_vector_store, setup_rag_pipeline

# Load environment variable for Groq API Key from Streamlit secrets
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# Set Streamlit page configuration
st.set_page_config(page_title="RAG Chatbot", layout="centered")

# App title and description
st.title("🤖 RAG Chatbot")
st.write("Ask anything based on the knowledge base.")

# Initialize the RAG pipeline
def init_pipeline():
    # Load and chunk data from file
    chunks = load_data("Data/data.txt")

    # Create vector store (embedding-based retrieval index)
    vector_store = create_vector_store(chunks)

    # Set up RAG pipeline using the vector store
    return setup_rag_pipeline(vector_store)

# Call the pipeline initialization
rag_chain = init_pipeline()

# Take user input (query)
query = st.text_input("Ask a question:")

# If the user submits a query
if query:
    # Run the RAG chain to get the answer
    result = rag_chain({"query": query})
    answer = result["result"]

    # Display the answer on the Streamlit app
    st.markdown("**Answer:**")
    st.write(answer)

    # Append the question and answer to a response log file
    with open("response_log.txt", "a", encoding="utf-8") as log_file:
        log_file.write(f"Q: {query}\nA: {answer}\n{'-'*50}\n")
