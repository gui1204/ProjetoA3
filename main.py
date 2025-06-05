from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from pymongo import MongoClient

app = FastAPI()

# URL MongoDB Atlas atualizada
MONGO_URI = "mongodb+srv://guilhermeCEO:rFgKo74oNCqE78rk@guicluster0.zfbpwpw.mongodb.net/chatbot_hb20?retryWrites=true&w=majority"

class Query(BaseModel):
    question: str

@app.on_event("startup")
def startup_event():
    global db, embeddings, mongo_collection

    # 1) Conectar MongoDB
    mongo_client = MongoClient(MONGO_URI)
    mongo_db = mongo_client['manual_database']
    mongo_collection = mongo_db['chunks']

    # 2) Ler PDF e extrair texto
    pdf_path = "manual.pdf"
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"

    # 3) Dividir texto em chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(full_text)

    # 4) Criar documentos (opcional)
    docs = [Document(page_content=t) for t in texts]

    # 5) Criar embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 6) Criar índice FAISS
    db = FAISS.from_texts(texts, embedding=embeddings)

    # 7) Inserir no MongoDB (limpar coleção antes)
    mongo_collection.delete_many({})
    for i, chunk in enumerate(texts):
        mongo_collection.insert_one({
            "chunk_id": i,
            "text": chunk
        })

@app.get("/")
def root_redirect():
    # Redireciona para /chat-hyundai
    return RedirectResponse(url="/chat-hyundai")

@app.get("/chat-hyundai")
def read_chat():
    return {"message": "Hyundai ChatBot - HB20 está on!!"}

@app.post("/query")
def query_manual(q: Query):
    query_embedding = embeddings.embed_query(q.question)
    results = db.similarity_search_by_vector(query_embedding, k=3)
    return {"results": [r.page_content for r in results]}

@app.get("/mongo_test")
def mongo_test():
    count = mongo_collection.count_documents({})
    cursor = mongo_collection.find({}, {"_id": 0, "chunk_id": 1, "text": 1}).limit(3)
    chunks = [{"chunk_id": doc["chunk_id"], "text": doc["text"]} for doc in cursor]
    return {"total_chunks": count, "sample_chunks": chunks}
