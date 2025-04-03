# server.py

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# Set your OpenAI API key (and ensure it's secure in production)
os.environ["OPENAI_API_KEY"] = "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"

# Create FastAPI app
app = FastAPI()

# Load documents from a folder (adjust path and glob as needed)
loader = DirectoryLoader(
    "./project_docs",  # Put your project documents in this folder
    glob="*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"}
)
documents = loader.load()
print(f"Loaded {len(documents)} documents from project_docs.")

# Split documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
split_documents = text_splitter.split_documents(documents)
print(f"Split documents into {len(split_documents)} chunks.")

# Create embeddings and vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(split_documents, embeddings, collection_name="project_docs")
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Set up a RetrievalQA chain using ChatOpenAI (GPT-3.5 Turbo)
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

# Define a request model
class QueryRequest(BaseModel):
    query: str

@app.post("/project-query")
async def project_query(request: QueryRequest):
    try:
        answer = qa_chain.run(request.query)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
