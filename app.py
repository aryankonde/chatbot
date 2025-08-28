import pickle
import numpy as np
import faiss
from flask import Flask, request, render_template, jsonify
from sentence_transformers import SentenceTransformer
import requests
import gspread 
from oauth2client.service_account import ServiceAccountCredentials 
from datetime import datetime 

# --- (Constants and model loading remain the same) ---
FAISS_PATH = 'faiss_index.bin'
META_PATH = 'faiss_meta.pkl'
MODEL_NAME = 'all-MiniLM-L6-v2'

print("[DEBUG] Loading FAISS index and metadata...")
index = faiss.read_index(FAISS_PATH)
with open(META_PATH, 'rb') as f:
    meta = pickle.load(f)
print(f"[DEBUG] Loaded {len(meta)} chunks from metadata.")
print("[DEBUG] Initializing embedding model...")
model = SentenceTransformer(MODEL_NAME)
print("[DEBUG] Embedding model loaded.")

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
GEMINI_API_KEY = "api_key" 

def get_gemini_answer(context, question):
    headers = {"Content-Type": "application/json"}
    system_prompt = (
        "You are an expert assistant for a Retrieval-Augmented Generation (RAG) application. "
        "Given the provided context, answer the user's question accurately and concisely. "
        "Base your answer *only* on the information within the context. "
        "If the answer is not present in the context, explicitly state: 'The answer is not available in the provided context.'"
    )
    prompt_text = f"Context:\n{context}\n\nQuestion:\n{question}"
    
    print("\n[DEBUG] --- Prompt Constructed for Gemini ---")
    print(prompt_text)
    print("-------------------------------------------\n")

    data = {"system_instruction": {"parts": [{"text": system_prompt}]}, "contents": [{"role": "user", "parts": [{"text": prompt_text}]}]}
    try:
        response = requests.post(f"{GEMINI_API_URL}?key={GEMINI_API_KEY}", headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        return f"Error from Gemini API: {e}"

def search_chunks(query, top_k=5):
    query_emb = model.encode([query])
    faiss.normalize_L2(query_emb)
    
    distances, indices = index.search(query_emb.astype(np.float32), top_k)
    
    results = [meta[idx] for idx in indices[0] if idx != -1 and idx < len(meta)]
    

    return results, distances[0]


#Gspread configuration and Flask app initialization
GSPREAD_SCOPE = ["https://spreadsheets.google.com/feeds", 'https://www.googleapis.com/auth/drive']
GSPREAD_CREDENTIALS = "credentials.json"
GOOGLE_SHEET_NAME = "Chatbot Support Tickets" 
app = Flask(__name__)


@app.route('/')
def home():
    """Renders the initial, empty chat page."""
    return render_template('index.html')

@app.route('/api/ask', methods=['POST'])
def api_ask():
    """Handles the chat logic with added debugging."""
    data = request.json
    user_query = data.get('query', '')
    if not user_query:
        return jsonify({'answer': 'Please provide a query.'}), 400

    print(f"\n[DEBUG] Received User Question: '{user_query}'")

    chunks, distances = search_chunks(user_query, top_k=3)
    
    print("\n[DEBUG] --- Top K Chunks Retrieved ---")
    if chunks:
        for i, chunk in enumerate(chunks):
            #FAISS with normalized vectors returns L2 distance. 
            #Convert L2 distance to Cosine Similarity: sim = 1 - (L2_dist^2 / 2)
            cosine_similarity = 1 - (distances[i]**2 / 2)
            print(f"  - Chunk {i+1}: Heading='{chunk['heading']}', Score={cosine_similarity:.4f}")
    else:
        print("  - No chunks found.")
    print("------------------------------------")
    

    if chunks:
        context = "\n\n---\n\n".join([f"Heading: {c['heading']}\n\n{c['content']}" for c in chunks])
        answer = get_gemini_answer(context, user_query)
    else:
        answer = "No relevant context found to answer your question."
        
    return jsonify({'answer': answer})


@app.route('/contact-support', methods=['POST'])
def contact_support():
    
    data = request.json
    name = data.get('name')
    email = data.get('email')
    question = data.get('question')

    if not all([name, email, question]):
        return jsonify({'status': 'error', 'message': 'All fields are required.'}), 400

    try:
        creds = ServiceAccountCredentials.from_json_keyfile_name(GSPREAD_CREDENTIALS, GSPREAD_SCOPE)
        client = gspread.authorize(creds)
        sheet = client.open(GOOGLE_SHEET_NAME).sheet1
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = [name, email, question, timestamp]
        sheet.append_row(row)
        return jsonify({'status': 'success', 'message': 'Your request has been submitted successfully.'})
    
    except FileNotFoundError:
        print("[ERROR] credentials.json not found.")
        return jsonify({'status': 'error', 'message': 'Server configuration error: credentials file not found.'}), 500
    except Exception as e:
        print(f"[ERROR] An error occurred with Google Sheets: {e}")
        return jsonify({'status': 'error', 'message': 'Could not submit your request due to a server error.'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)