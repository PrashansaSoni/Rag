import os
from langchain_community.document_loaders import TextLoader  
from langchain.text_splitter import CharacterTextSplitter    
from langchain_community.vectorstores import FAISS         
from langchain.embeddings import OllamaEmbeddings           
from langchain_core.runnables import Runnable                 
from langchain.prompts import PromptTemplate                 
from langchain_groq import ChatGroq                         
from langchain.chains import RetrievalQA                     

# Function to load and split data from a text file
def load_data(file_path: str):
    loader = TextLoader(file_path)  
    documents = loader.load()      
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)  
    return text_splitter.split_documents(documents)  

# Function to create a vector store (FAISS) 
def create_vector_store(chunks):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")  
    return FAISS.from_documents(chunks, embeddings)          

# Function to set up the full RAG pipeline
def setup_rag_pipeline(vector_store):
    retriever = vector_store.as_retriever()  

    # Define the prompt format used by the LLM
    prompt_template = PromptTemplate(
        template="""
        You are an expert chatbot. Use the following context to answer the question.
        If you don't know the answer, say so honestly.

        Context:
        {context}

        Question:
        {question}

        Answer:""",
        input_variables=["context", "question"] 
    )

    # Instantiate the LLM from Groq with the specified model
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),  
        model_name="llama3-8b-8192"            
    )

    # Create a RetrievalQA chain combining the LLM, retriever, and custom prompt
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,                                
        retriever=retriever,                     
        return_source_documents=True,           
        chain_type_kwargs={"prompt": prompt_template} 
    )

    return rag_chain  
