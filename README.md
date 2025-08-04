# 🤖 ChatBot Hyundai HB20 com IA

Este projeto é um assistente virtual para usuários do Hyundai HB20. Ele utiliza Inteligência Artificial para responder perguntas com base no manual do carro, carregado em formato PDF. Toda a experiência é acessível via uma interface web simples e responsiva.

## 🔧 Tecnologias Utilizadas

- **FastAPI** — API backend rápida e moderna
- **Google Gemini 1.5 Flash** — modelo LLM para geração de respostas
- **LangChain + FAISS** — para buscas vetoriais eficientes nos textos
- **MongoDB Atlas** — armazenamento dos chunks indexados
- **HuggingFace Embeddings** — transformações semânticas de texto
- **PDF Parsing** — extração do conteúdo do manual
- **HTML/CSS Vanilla** — interface web responsiva com background do HB20

## 🚗 Funcionalidades

- Carrega e processa o manual em PDF
- Divide o conteúdo em partes ("chunks") e salva no MongoDB
- Realiza buscas semânticas sobre os chunks mais relevantes
- Gera respostas com base exclusivamente no conteúdo do manual
- Interface web com campo de pergunta/resposta

## 📦 Instalação

1. **Clone o projeto**
```bash
https://github.com/gui1204/ProjetoA3
