# Import necessary packages
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import hub
# install huggingface using  pip install langchain-huggingface
from langchain_huggingface import HuggingFaceEmbeddings
# Define the embeddings to be used
other_embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")

# Load and split the PDF
pdfloader = PyPDFLoader("It-ends-with-us.pdf")
pdfpages = pdfloader.load_and_split()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

split_docs = []
for page in pdfpages:
    split_docs.extend(text_splitter.split_text(page.page_content))

# Using chromeDB
vector_store = Chroma.from_texts(split_docs, other_embeddings)

# Define the LLM model ie llama or qwen
llm = Ollama(
    model="qwen2:0.5b"
)

retriever = vector_store.as_retriever()
print("pulling the prompt from hub ")
prompt = hub.pull("rlm/rag-prompt")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("working on it through the chain ......")
print(rag_chain.invoke("Who is lily, describe about her in 100 words"))


# This is for similarity search

# retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 6})
# retrieved_docs = retriever.invoke("Who is lily ?")
# print(retrieved_docs)
