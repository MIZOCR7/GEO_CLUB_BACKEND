import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

BASE_DIR = Path(__file__).parent
dotenv_path = BASE_DIR / ".env"

if dotenv_path.exists():
    load_dotenv(dotenv_path)

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    print("CRITICAL ERROR: GROQ_API_KEY not found.")
    sys.exit(1)

client = Groq(api_key=groq_api_key)

CLUB_INFO = {
    "name": "Geology Club",
    "school": "STEM High School For Boys",
    "president": "Ahmed Mohamed Abd Altuab",
    "president_phone": "+20 123 456 7890",
    "vice_presidents": [{"name": "Amir", "phone": "+20 123 456 7891"}, {"name": "Dosouky", "phone": "+20 123 456 7892"}],
}

DB_FOLDER_NAME = "geology_club_final_db"
DB_PATH = BASE_DIR / DB_FOLDER_NAME
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_db = None

# ==========================================
# 2. LIFESPAN (STARTUP LOGIC)
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_db
    print("API Starting... Loading Knowledge Base...")
    
    if os.path.exists(DB_PATH):
        try:
            vector_db = FAISS.load_local(str(DB_PATH), embedding_model, allow_dangerous_deserialization=True)
            print("Database loaded successfully.")
        except Exception as e:
            print(f"⚠️ Error loading DB: {e}")
    else:
        print("No database found! Please run the builder script first.")
    
    yield
    print("API Shutting down...")

# ==========================================
# 3. APP DEFINITION
# ==========================================
app = FastAPI(title="Geology Club API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 4. DATA MODELS
# ==========================================
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[Message] = []

class ChatResponse(BaseModel):
    response: str

# ==========================================
# 5. CORE LOGIC
# ==========================================
def get_professor_response(user_input: str, history: List[Message]) -> str:
    context_str = ""
    if vector_db:
        try:
            results = vector_db.similarity_search(user_input, k=6)
            seen = set()
            unique = []
            for doc in results:
                key = doc.page_content[:100]
                if key not in seen:
                    seen.add(key)
                    unique.append(doc)
            context_str = "\n\n".join([
                f"[Source: {doc.metadata.get('source_type')} | Page: {doc.metadata.get('page_number')}]\n{doc.page_content}"
                for doc in unique
            ])
        except Exception as e:
            print(f"DB Search Error: {e}")

    system_instruction = f"""You are the AI Professor for {CLUB_INFO['name']} at {CLUB_INFO['school']}.

TEXTBOOK CONTEXT:
{context_str}

YOUR ROLE:
- Answer geology questions using the TEXTBOOK CONTEXT above
- If the context lacks the answer, say so — never make up information
- Cite every factual claim: [Page X] or [Source: type | Page: X]
- Format with ## headings, **bold** key terms, - bullet lists

RESPONSE STYLE:
- Be concise but deep — include mechanisms, causes, effects
- End with a ### References section listing every page cited
- Never ask follow-up questions or suggest what to do next"""

    try:
        messages = [{"role": "system", "content": system_instruction}]

        for msg in history[-6:]:
            role = "user" if msg.role in ["Student", "user"] else "assistant"
            messages.append({"role": role, "content": msg.content})

        messages.append({"role": "user", "content": user_input})

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.3,
            max_tokens=3000,
            top_p=0.95,
        )

        result = response.choices[0].message.content
        if not result or result.strip() == "":
            return "The AI professor couldn't generate a response. Please try rephrasing your question."
        return result

    except Exception as e:
        error_msg = str(e).lower()
        if "content_filter" in error_msg or "safety" in error_msg:
            return "Your message was flagged by our safety filter. Please try asking your question differently."
        elif "timeout" in error_msg or "deadline" in error_msg:
            return "The response took too long. Please try again."
        elif "401" in error_msg or "authentication" in error_msg or "unauthenticated" in error_msg:
            return "System authentication error. Please check the API key."
        elif "quota" in error_msg or "rate_limit" in error_msg:
            return "API quota exceeded or Too Many Requests. Please wait a minute and try again."
        else:
            return f"Connection error: {str(e)}"

# ==========================================
# 6. ENDPOINTS
# ==========================================
@app.get("/")
@app.get("/api/health")
def health_check():
    return {"status": "running", "club": CLUB_INFO["name"]}

@app.get("/api/chat")
def chat_get(message: str = "", history: str = ""):
    if not message:
        raise HTTPException(status_code=400, detail="Message parameter is required")
    try:
        bot_reply = get_professor_response(message, [])
        return {"response": bot_reply, "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat", response_model=ChatResponse)
@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    if not request.message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    try:
        bot_reply = get_professor_response(request.message, request.history)
        return {"response": bot_reply, "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
