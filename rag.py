from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

pdfloader = PyPDFLoader("/home/prashansa-soni/Downloads/It ends with us.pdf")
pdfpages = pdfloader.load_and_split()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)



split_docs = []
for page in pdfpages:
    split_docs.extend(text_splitter.split_text(page.page_content))



embeddings = OpenAIEmbeddings( )


vector_store = FAISS.from_texts(split_docs, embeddings)


model = Ollama()

rag_chain = RetrievalQA(llm=model, retriever=vector_store.as_retriever())

query = "Who is Lily?"


# Get the answer
result = rag_chain(query)
print(result)





