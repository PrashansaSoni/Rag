from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
# Install these packages through pip
from langchain_ollama import OllamaEmbeddings
from langchain_community.embeddings import SentenceTransformerEmbeddings

# If you want to use llama3 embeddings 
ollama_embeddings = OllamaEmbeddings(model="llama3")

# If you want to use other embeddings like 
other_embeddings = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")

pdfloader = PyPDFLoader("/home/prashansa-soni/Downloads/It ends with us.pdf")
pdfpages = pdfloader.load_and_split()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)



split_docs = []
for page in pdfpages:
    split_docs.extend(text_splitter.split_text(page.page_content))


# Remove this line and use the above lines instead
embeddings = OpenAIEmbeddings( )

# Change the embeddings function name below to as defined earlier, ie ollama_embeddings or other_embeddings
vector_store = FAISS.from_texts(split_docs, embeddings)


model = Ollama()

rag_chain = RetrievalQA(llm=model, retriever=vector_store.as_retriever())

query = "Who is Lily?"


# Get the answer
result = rag_chain(query)
print(result)





