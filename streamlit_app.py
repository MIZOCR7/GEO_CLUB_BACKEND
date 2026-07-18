import os
from pathlib import Path
from typing import List
import streamlit as st

st.set_page_config(page_title="Geology Club API", page_icon="🌋", layout="centered")

from groq import Groq

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).parent

groq_api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")

if not groq_api_key:
    st.error("CRITICAL ERROR: GROQ_API_KEY not found in Environment or Secrets.")
    st.stop()

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

# ==========================================
# 2. LIGHTWEIGHT VECTOR DB RETRIEVAL
# ==========================================
def get_professor_response(user_input: str, history: list) -> str:
    context_str = ""
    
    if os.path.exists(DB_PATH):
        try:
            from langchain_community.vectorstores import FAISS
            from langchain_core.embeddings import Embeddings
            
            import google.generativeai as genai

            class GeminiEmbeddingsWrapper(Embeddings):
                def embed_documents(self, texts: List[str]) -> List[List[float]]:
                    return [genai.embed_content(model="models/embedding-001", content=t)["embedding"] for t in texts]
                def embed_query(self, text: str) -> List[float]:
                    return genai.embed_content(model="models/embedding-001", content=text)["embedding"]
            
            vector_db = FAISS.load_local(str(DB_PATH), GeminiEmbeddingsWrapper(), allow_dangerous_deserialization=True)
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

## TEXTBOOK CONTEXT
{context_str}

## YOUR ROLE
- Answer geology questions **exclusively** using the provided TEXTBOOK CONTEXT above
- If the context lacks the answer, say "This isn't covered in the textbook" — never make up information
- Cite **every** factual claim with its source: [Page X] or [Source: type | Page: X]
- Use **bold** for key geology terms, `important` concepts, and definitions
- For explanations: dig deep into mechanisms, processes, and cause-effect chains
- For definitions: state concisely then expand with textbook details

## FORMATTING RULES (strict)
Always structure your response using EXACTLY this markdown:

## Main Topic
**Key Term** — brief definition. [Page X]

### Subtopic / Mechanism
- **Point one** with detailed textbook explanation. [Page X]
- **Point two** — causal chain, consequences, related concepts. [Page X]
- Use `sub-concepts` sparingly for emphasis.

### Key Takeaways
1. First takeaway with page reference.
2. Second takeaway with page reference.

### References
- [Page X] — Topic or definition from this page
- [Page Y] — Topic or definition from this page

## BEHAVIOR RULES
- **Be concise** — answer directly in 3-6 paragraphs max
- **Be deep** — when explaining, include mechanisms, causes, effects, and interconnections from the textbook
- **Always** end every response with a `### References` section listing every page cited
- Never ask follow-up questions or suggest what the student should do next
- Never mention "I'd be happy to" or "Let me know if"
- Never include meta-commentary about your response
- If asked to evaluate an answer: state if correct/incorrect, explain why using textbook, cite the page
- If asked to generate a quiz: output questions with **bold** key terms and [Page X] references"""

    try:
        messages = [{"role": "system", "content": system_instruction}]

        for msg in history[-6:]:
            role = "user" if msg.get("role") in ["Student", "user"] else "assistant"
            messages.append({"role": role, "content": msg.get("content")})

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
            return "System authentication error. Please check the API key in Secrets."
        elif "quota" in error_msg or "rate_limit" in error_msg:
            return "API quota exceeded or Too Many Requests. Please wait a minute and try again."
        else:
            return f"Connection error: {str(e)}"

# ==========================================
# 3. NATIVE STREAMLIT API ROUTING
# ==========================================
query_params = st.query_params

if "api" in query_params:
    endpoint = query_params.get("api")

    if endpoint == "health":
        st.json({"status": "running", "club": CLUB_INFO["name"]})
        st.stop()

    elif endpoint == "chat":
        user_message = query_params.get("message", "")
        if not user_message:
            st.json({"error": "Message parameter is required"})
            st.stop()

        bot_reply = get_professor_response(user_message, history=[])
        st.json({"response": bot_reply})
        st.stop()

# ==========================================
# 4. DASHBOARD INTERFACE
# ==========================================
st.title("Geology Club - AI Professor")
st.success("Your Backend API is Live and Fully Operational!")

st.markdown("""
### API Endpoints for Frontend (Vercel):
- **Health Check:** `https://geoclub-backend.streamlit.app/?api=health`
- **Chat:** `https://geoclub-backend.streamlit.app/?api=chat&message=YOUR_QUESTION`
""")

st.divider()
st.subheader("Ask the AI Professor")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Type your geology question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = get_professor_response(prompt, st.session_state.messages[:-1])
        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})

st.info("When sending requests from the frontend, make sure to URL-encode the message if it contains spaces or special characters.")