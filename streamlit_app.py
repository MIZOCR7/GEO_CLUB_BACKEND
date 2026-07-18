import os
import sys
from pathlib import Path
from typing import List
import streamlit as st

# 1. إعداد الصفحة كأول أمر إلزامي في Streamlit
st.set_page_config(page_title="Geology Club API", page_icon="🌋", layout="centered")

import google.generativeai as genai
from pydantic import BaseModel

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).parent

# جلب المفتاح بأمان من السيرفر أو الـ Secrets
gemini_api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")

if not gemini_api_key:
    st.error("❌ CRITICAL ERROR: GEMINI_API_KEY not found in Environment or Secrets.")
    st.stop()

# إعداد الـ Gemini API
genai.configure(api_key=gemini_api_key)

# استخدمنا الموديل المستقر والمثالي لتجنب مشاكل الـ Quota المزعجة
model = genai.GenerativeModel("gemini-2.0-flash")

CLUB_INFO = {
    "name": "Geology Club",
    "school": "STEM High School For Boys",
    "president": "Ahmed Mohamed Abd Altuab",
    "president_phone": "+20 123 456 7890",
    "vice_presidents": [{"name": "Amir", "phone": "+20 123 456 7891"}, {"name": "Dosouky", "phone": "+20 123 456 7892"}],
}

DB_FOLDER_NAME = "geology_club_final_db"
DB_PATH = BASE_DIR / DB_FOLDER_NAME

# ==========================================
# 2. LIGHTWEIGHT VECTOR DB RETRIEVAL
# ==========================================
def get_professor_response(user_input: str, history: list) -> str:
    context_str = ""
    
    # تحميل قاعدة البيانات بذكاء وبدون استهلاك الرام بموديلات محليّة ضخمة
    if os.path.exists(DB_PATH):
        try:
            from langchain_community.vectorstores import FAISS
            from langchain_core.embeddings import Embeddings
            
            # غلاف ذكي يوجه حسابات الفيكتور لـ Gemini مباشرة لإنقاذ السيرفر من التجميد
            class GeminiEmbeddingsWrapper(Embeddings):
                def embed_documents(self, texts: List[str]) -> List[List[float]]:
                    return [genai.embed_content(model="models/embedding-001", content=t)["embedding"] for t in texts]
                def embed_query(self, text: str) -> List[float]:
                    return genai.embed_content(model="models/embedding-001", content=text)["embedding"]
            
            vector_db = FAISS.load_local(str(DB_PATH), GeminiEmbeddingsWrapper(), allow_dangerous_deserialization=True)
            results = vector_db.similarity_search(user_input, k=4)
            context_str = "\n\n".join([
                f"[Source: {doc.metadata.get('source_type')} | Page: {doc.metadata.get('page_number')}]\n{doc.page_content}"
                for doc in results
            ])
        except Exception as e:
            print(f"DB Search Error: {e}")

    # الـ System Prompt الاحترافي الخاص بك كما هو تماماً
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

    try:
        full_prompt = system_instruction + "\n\n--- CONVERSATION ---\n"
        
        for msg in history[-6:]:
            role = "Student" if msg.get("role") in ["Student", "user"] else "Professor"
            full_prompt += f"{role}: {msg.get('content')}\n"
        
        full_prompt += f"Student: {user_input}\n\nProfessor:"
        
        response = model.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.2,
                max_output_tokens=2000,
                top_p=0.9,
            ),
        )
        
        result = response.text
        if not result or result.strip() == "":
            return "⚠️ The AI professor couldn't generate a response. Please try rephrasing your question."
        return result

    except Exception as e:
        error_msg = str(e).lower()
        if "content_filter" in error_msg or "safety" in error_msg:
            return "⚠️ Your message was flagged by our safety filter. Please try asking your question differently."
        elif "timeout" in error_msg or "deadline" in error_msg:
            return "⚠️ The response took too long. Please try again."
        elif "quota" in error_msg or "429" in error_msg or "rate_limit" in error_msg:
            return "⚠️ API quota exceeded or Too Many Requests. Please wait a minute and try again."
        else:
            return f"⚠️ Connection error: {str(e)}"

# ==========================================
# 3. NATIVE STREAMLIT API ROUTING
# ==========================================
query_params = st.query_params

if "api" in query_params:
    endpoint = query_params.get("api")
    
    # 1. Endpoint: Health Check (المسار الجديد: /?api=health)
    if endpoint == "health":
        st.json({"status": "running", "club": CLUB_INFO["name"]})
        st.stop()
        
    # 2. Endpoint: Chat (المسار الجديد: /?api=chat&message=سؤالك_هنا)
    elif endpoint == "chat":
        user_message = query_params.get("message", "")
        if not user_message:
            st.json({"error": "Message parameter is required"})
            st.stop()
            
        # معالجة الرد وإرساله كـ JSON نقي للـ Frontend
        bot_reply = get_professor_response(user_message, history=[])
        st.json({"response": bot_reply})
        st.stop()

# ==========================================
# 4. DASHBOARD INTERFACE
# ==========================================
st.title("🌋 STEM Geology Club - API Server")
st.success("🚀 Your Backend API is Live and Fully Operational!")

st.markdown("""
### 🔗 روابط الـ API الرسمية لربطها بالـ Frontend (Vercel):
* **Health Check Endpoint:**  
  `https://geoclub-backend.streamlit.app/?api=health`
* **Chat Endpoint:**  
  `https://geoclub-backend.streamlit.app/?api=chat&message=YOUR_QUESTION`
""")

st.info("💡 ملاحظة للـ Frontend: عند إرسال الأسئلة، تأكد من عمل Encode للرابط (URL Encode) إذا كان السؤال يحتوي على مسافات أو لغة عربية.")