from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from pymongo import MongoClient
import certifi
import google.generativeai as genai
import os

# === CONFIG GOOGLE GEMINI ===
os.environ["GOOGLE_API_KEY"] = "AIzaSyC8JT-3TWaD0rEUgcthN77zsGicumEXd98"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Use um modelo válido listado na API do Google
model = genai.GenerativeModel('models/text-bison-001')

# === APP SETUP ===
app = FastAPI()
templates = Jinja2Templates(directory="templates")

MONGO_URI = "mongodb+srv://guilhermeCEO:rFgKo74oNCqE78rk@guicluster0.zfbpwpw.mongodb.net/chatbot_hb20?retryWrites=true&w=majority"

class Query(BaseModel):
    pergunta: str

@app.on_event("startup")
def startup_event():
    global db, embeddings, mongo_collection

    mongo_client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
    mongo_db = mongo_client['manual_database']
    mongo_collection = mongo_db['chunks']

    pdf_path = "manual.pdf"
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(full_text)

    docs = [Document(page_content=t) for t in texts]
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_texts(texts, embedding=embeddings)

    mongo_collection.delete_many({})
    for i, chunk in enumerate(texts):
        mongo_collection.insert_one({
            "chunk_id": i,
            "text": chunk
        })

@app.get("/")
def root_redirect():
    return RedirectResponse(url="/chat-hyundai")

@app.get("/chat-hyundai")
def read_chat(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/query")
def query_manual(q: Query):
    query_embedding = embeddings.embed_query(q.pergunta)
    results = db.similarity_search_by_vector(query_embedding, k=3)
    contexto = "\n\n".join([r.page_content for r in results])

    system_prompt = f"""
Você é um agente de inteligência artificial especializado em suporte ao usuário com base no manual de um carro. Seu papel é responder perguntas somente com base nas informações presentes no contexto fornecido abaixo.

Siga estas regras com rigor:

1. Utilize apenas as informações contidas no contexto fornecido entre {{contexto}}.

2. Não invente respostas. Se a informação solicitada não estiver presente no contexto, responda:
"Essa informação não está disponível no conteúdo fornecido do manual."

3. Seja claro, objetivo e técnico quando necessário. Evite rodeios.

4. Se a resposta estiver no contexto, cite exatamente as instruções relevantes, sem alterações.

---

Contexto:
{contexto}

Pergunta do usuário:
{q.pergunta}

Sua resposta:
""".strip()

    try:
        resposta_modelo = model.generate_content(system_prompt)
        texto_resposta = resposta_modelo.text.strip()

        if not texto_resposta:
            return {"resposta": "Nenhuma resposta encontrada no conteúdo do modelo."}
        return {"resposta": texto_resposta}
    except Exception as e:
        return {"error": str(e)}

@app.get("/mongo_test")
def mongo_test():
    count = mongo_collection.count_documents({})
    cursor = mongo_collection.find({}, {"_id": 0, "chunk_id": 1, "text": 1}).limit(3)
    chunks = [{"chunk_id": doc["chunk_id"], "text": doc["text"]} for doc in cursor]
    return {"total_chunks": count, "sample_chunks": chunks}
