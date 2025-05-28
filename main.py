from fastapi import FastAPI
from pydantic import BaseModel
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Chatbot do HB20 está online!"}
class Query(BaseModel):
    question: str

@app.on_event("startup")
def startup_event():
    global db, embeddings

    # Ler PDF
    pdf_path = "manual.pdf"
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"

    # Dividir texto em pedaços
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(full_text)

    # Criar documentos para LangChain (opcional se usar só textos)
    docs = [Document(page_content=t) for t in texts]

    # Criar embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Criar índice FAISS
    db = FAISS.from_texts(texts, embedding=embeddings)

@app.post("/query")
def query_manual(q: Query):
    # Criar embedding da pergunta
    query_embedding = embeddings.embed_query(q.question)

    # Buscar textos similares no FAISS
    results = db.similarity_search_by_vector(query_embedding, k=3)

    return {"results": [r.page_content for r in results]}
