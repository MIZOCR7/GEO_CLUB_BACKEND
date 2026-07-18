import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List, Optional
from dotenv import load_dotenv

# FastAPI Imports
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# AI Imports
from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).parent
dotenv_path = BASE_DIR / ".env"

if dotenv_path.exists():
    load_dotenv(dotenv_path)

github_token = os.getenv("GITHUB_TOKEN")
if not github_token:
    print("❌ CRITICAL ERROR: GITHUB_TOKEN not found.")
    sys.exit(1)

client = OpenAI(
    base_url="https://models.github.ai/inference",
    api_key=github_token,
)

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
    print("🚀 API Starting... Loading Knowledge Base...")
    
    if os.path.exists(DB_PATH):
        try:
            vector_db = FAISS.load_local(str(DB_PATH), embedding_model, allow_dangerous_deserialization=True)
            print("✅ Database loaded successfully.")
        except Exception as e:
            print(f"⚠️ Error loading DB: {e}")
    else:
        print("⚠️ No database found! Please run the builder script first.")
    
    yield
    print("🛑 API Shutting down...")

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
# 5. CORE LOGIC (SAFE VERSION)
# ==========================================
def get_professor_response(user_input: str, history: List[Message]) -> str:
    # 1. Retrieve Context (RAG)
    context_str = ""
    if vector_db:
        try:
            results = vector_db.similarity_search(user_input, k=4)
            context_str = "\n\n".join([
                f"[Source: {doc.metadata.get('source_type')} | Page: {doc.metadata.get('page_number')}]\n{doc.page_content}"
                for doc in results
            ])
        except Exception as e:
            print(f"DB Search Error: {e}")

    # 2. System Prompt (Professional, Accurate, No Task Suggestions)
    system_instruction = f"""
    You are the AI Professor for {CLUB_INFO['name']} at {CLUB_INFO['school']}.

    TEXTBOOK CONTEXT:
    {context_str}

    **YOUR ROLE:**
    - Provide clear, accurate geology explanations based on textbook content
    - Evaluate exam answers with precision
    - Generate high-quality quizzes from course material
    - Always cite Source and Page number when available

    **RESPONSE STYLE:**
    - Use clear, structured formatting (bullet points, numbered lists)
    - Explain concepts step-by-step for student understanding
    - Avoid asking what to do next - just provide the answer
    - Be professional and academically rigorous
    - Do NOT mention tasks, suggestions, or what students should do

    **FOR QUIZ ANSWERS:**
    - Evaluate student answers with full explanations
    - Provide the correct answer with reasoning from the textbook
    - Give immediate feedback without asking follow-up questions

    **FOR EXPLANATIONS:**
    - Structure with headings and sub-points
    - Include page references: [Page X]
    - Make it student-friendly but academically accurate
    """

    # 3. Build Safe Message Chain (Prevents "Jailbreak" Error)
    messages = [{"role": "system", "content": system_instruction}]
    
    # Add History
    for msg in history[-6:]:
        # Map frontend roles to OpenAI roles
        role = "user" if msg.role in ["Student", "user"] else "assistant"
        messages.append({"role": role, "content": msg.content})
    
    # Add Current Question
    messages.append({"role": "user", "content": user_input})

    try:
        response = client.chat.completions.create(
            messages=messages,
            model="gpt-4o",
            temperature=0.2,  # Lower for better accuracy
            max_tokens=2000,
            top_p=0.9,
        )
        result = response.choices[0].message.content
        if not result or result.strip() == "":
            return "⚠️ The AI professor couldn't generate a response. Please try rephrasing your question."
        return result
    except Exception as e:
        error_msg = str(e).lower()
        if "content_filter" in error_msg or "safety" in error_msg:
            return "⚠️ Your message was flagged by our safety filter. Please try asking your question differently."
        elif "timeout" in error_msg or "deadline" in error_msg:
            return "⚠️ The response took too long. Please try again."
        elif "401" in error_msg or "authentication" in error_msg:
            return "⚠️ System authentication error. Please contact the administrator."
        else:
            return f"⚠️ Connection error. Please try again in a moment."

# ==========================================
# 6. ENDPOINTS
# ==========================================
@app.get("/")
def health_check():
    return {"status": "running", "club": CLUB_INFO["name"]}

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    if not request.message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    bot_reply = get_professor_response(request.message, request.history)
    
    return ChatResponse(response=bot_reply)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
