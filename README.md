---
title: Geology Club AI Backend
emoji: 🌋
colorFrom: #C29B6D
colorTo: #2B1B10
sdk: docker
pinned: false
---

# Geology Club AI Backend

FastAPI backend for the Geology Club AI Professor — a RAG-powered chatbot using Groq LLM + FAISS vector search.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` or `/api/health` | Health check |
| GET | `/api/chat?message=...` | Chat via query param |
| POST | `/api/chat` | Chat with JSON body `{"message": "...", "history": []}` |

## Environment Variables

```
GROQ_API_KEY=your_groq_api_key
```

## Run Locally

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add your GROQ_API_KEY
uvicorn main:app --host 0.0.0.0 --port 8000
```
