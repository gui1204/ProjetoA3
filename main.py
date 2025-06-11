from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pypdf import PdfReader
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

# Use um modelo válido do Google Gemini
model = genai.GenerativeModel('gemini-1.5-flash')

MONGO_URI = "mongodb+srv://guilhermeCEO:rFgKo74oNCqE78rk@guicluster0.zfbpwpw.mongodb.net/chatbot_hb20?retryWrites=true&w=majority"

# URI alternativo com configurações SSL
MONGO_URI_ALT = "mongodb+srv://guilhermeCEO:rFgKo74oNCqE78rk@guicluster0.zfbpwpw.mongodb.net/chatbot_hb20?retryWrites=true&w=majority&ssl=true&ssl_cert_reqs=CERT_NONE"

# Variáveis globais
db = None
embeddings = None
mongo_collection = None

class Query(BaseModel):
    pergunta: str

def try_mongodb_connection():
    """Tenta conectar ao MongoDB usando diferentes estratégias"""
    
    strategies = [
        {
            "name": "Estratégia 1: TLS Básico",
            "config": {
                "uri": MONGO_URI,
                "options": {
                    "tls": True,
                    "serverSelectionTimeoutMS": 5000,
                    "connectTimeoutMS": 10000,
                    "socketTimeoutMS": 10000,
                }
            }
        },
        {
            "name": "Estratégia 2: TLS com Certificados Flexíveis", 
            "config": {
                "uri": MONGO_URI,
                "options": {
                    "tls": True,
                    "tlsAllowInvalidCertificates": True,
                    "tlsAllowInvalidHostnames": True,
                    "serverSelectionTimeoutMS": 5000,
                    "connectTimeoutMS": 10000,
                }
            }
        },
        {
            "name": "Estratégia 3: Sem TLS (teste)",
            "config": {
                "uri": MONGO_URI.replace("mongodb+srv://", "mongodb://").replace("?retryWrites=true&w=majority", ""),
                "options": {
                    "serverSelectionTimeoutMS": 5000,
                    "connectTimeoutMS": 10000,
                }
            }
        },
        {
            "name": "Estratégia 4: TLS com CA Bundle",
            "config": {
                "uri": MONGO_URI,
                "options": {
                    "tls": True,
                    "tlsCAFile": certifi.where(),
                    "serverSelectionTimeoutMS": 5000,
                    "connectTimeoutMS": 10000,
                }
            }
        },
        {
            "name": "Estratégia 5: SSL Legacy",
            "config": {
                "uri": MONGO_URI,
                "options": {
                    "ssl": True,
                    "serverSelectionTimeoutMS": 5000,
                    "connectTimeoutMS": 10000,
                }
            }
        }
    ]
    
    for strategy in strategies:
        try:
            print(f"🔄 Tentando: {strategy['name']}")
            
            client = MongoClient(
                strategy['config']['uri'],
                **strategy['config']['options']
            )
            
            # Teste a conexão
            client.admin.command('ping')
            print(f"✅ {strategy['name']} - Sucesso!")
            
            return client
            
        except Exception as e:
            print(f"❌ {strategy['name']} - Falhou: {str(e)[:100]}...")
            continue
    
    print("❌ Todas as estratégias de conexão falharam")
    return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global db, embeddings, mongo_collection

    # Tentativa de conexão com MongoDB usando múltiplas estratégias
    print("🔗 Tentando conectar ao MongoDB...")
    mongo_client = try_mongodb_connection()
    
    if mongo_client:
        try:
            mongo_db = mongo_client['manual_database']
            mongo_collection = mongo_db['chunks']
            mongo_collection.delete_many({})
            print("✅ MongoDB configurado com sucesso!")
        except Exception as e:
            print(f"⚠️  Erro ao configurar MongoDB: {e}")
            mongo_collection = None
    else:
        print("🔄 Continuando sem MongoDB...")
        mongo_collection = None

    pdf_path = "manual.pdf"
    if not os.path.exists(pdf_path):
        print(f"⚠️  Arquivo {pdf_path} não encontrado!")
        texts = ["Texto de exemplo para teste do sistema."]
    else:
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

    if mongo_collection is not None:
        try:
            for i, chunk in enumerate(texts):
                mongo_collection.insert_one({
                    "chunk_id": i,
                    "text": chunk
                })
            print(f"✅ {len(texts)} chunks salvos no MongoDB!")
        except Exception as e:
            print(f"⚠️  Erro ao salvar no MongoDB: {e}")
    
    print("🚀 Aplicação iniciada com sucesso!")
    yield
    # Shutdown (se necessário)

# === APP SETUP ===
app = FastAPI(lifespan=lifespan)

# Monta a pasta estática para servir arquivos como imagens
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

@app.get("/")
def root_redirect():
    return RedirectResponse(url="/chat-hyundai")

@app.get("/chat-hyundai")
def read_chat(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/query")
def query_manual(q: Query):
    query_embedding = embeddings.embed_query(q.pergunta)
    results = db.similarity_search_by_vector(query_embedding, k=5)
    print('Resultados da busca:')
    for i, result in enumerate(results):
        print(f'Resultado {i+1}: {result}')
    contexto = "\n\n".join([r.page_content for r in results])

    system_prompt = f"""
Você é um agente de inteligência artificial especializado em suporte ao usuário com base no manual de um carro. Seu papel é responder perguntas somente com base nas informações presentes no contexto fornecido abaixo.

Siga estas regras com rigor:

1. Utilize apenas as informações contidas no contexto.

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
    if mongo_collection is None:
        return {"error": "MongoDB não está conectado", "total_chunks": 0, "sample_chunks": []}
    
    try:
        count = mongo_collection.count_documents({})
        cursor = mongo_collection.find({}, {"_id": 0, "chunk_id": 1, "text": 1}).limit(3)
        chunks = [{"chunk_id": doc["chunk_id"], "text": doc["text"]} for doc in cursor]
        return {"total_chunks": count, "sample_chunks": chunks}
    except Exception as e:
        return {"error": f"Erro ao acessar MongoDB: {str(e)}", "total_chunks": 0, "sample_chunks": []}
