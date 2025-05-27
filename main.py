import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

load_dotenv()
app = FastAPI()

# Carrega PDF
loader = PyPDFLoader("manual.pdf")
pages = loader.load()

# Divide o texto em pedaços
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(pages)

# Cria embeddings com FAISS (local, rápido e gratuito)
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(docs, embeddings)

# Configura modelo LLM + QA chain
llm = OpenAI(temperature=0)
qa_chain = load_qa_chain(llm, chain_type="stuff")

class Question(BaseModel):
    question: str

@app.post("/ask")
async def ask(q: Question):
    docs_similares = db.similarity_search(q.question)
    resposta = qa_chain.run(input_documents=docs_similares, question=q.question)
    return {"answer": resposta}
