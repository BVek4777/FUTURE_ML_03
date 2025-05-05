import os
import json
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer, util
import joblib
from utils.load_data import data_loader
import re

# Configuration 
MODEL_DIR = 'model'
EMBEDDINGS_PATH = os.path.join(MODEL_DIR, 'embeddings.joblib')
DATA_PATH = os.path.join(MODEL_DIR, 'data.joblib')

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Global Variables 
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = None
questions = None
answers = None
df = None
model_trained = False

# Function to check for exact matches in user query like hi or bye
def is_exact_match(user_query, response_dict):
    for key in response_dict:
        if re.search(rf'\b{re.escape(key)}\b', user_query):
            return response_dict[key]
    return None
# --- Flask App Initialization ---
app = Flask(__name__)

# Load Model from Disk 
def load_model():
    global embeddings, df, questions, answers, model_trained
    try:
        embeddings = joblib.load(EMBEDDINGS_PATH)
        df = joblib.load(DATA_PATH)
        questions = df['question'].tolist()
        answers = df['answer'].tolist()
        model_trained = True
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Model load failed: {e}")
        model_trained = False

# Train the Sentence Transformer Model 
def train_model():
    global embeddings, df, questions, answers, model_trained
    try:
        df_raw = data_loader()
        records = []
        for entry in df_raw['qa']:
            if isinstance(entry, str):
                try:
                    entry = json.loads(entry)
                except json.JSONDecodeError:
                    continue
            for item in entry.get('knowledge', []):
                q = item.get('customer_summary_question', '').strip()
                a = item.get('agent_summary_solution', '').strip()
                if q:
                    records.append({'question': q, 'answer': a or 'No solution provided.'})

        if not records:
            return False, "Training failed: No valid QA pairs found."

        df_cleaned = pd.DataFrame(records)
        df = df_cleaned
        questions = df['question'].tolist()
        answers = df['answer'].tolist()
        embeddings = model.encode(questions, convert_to_tensor=True)

        joblib.dump(embeddings, EMBEDDINGS_PATH)
        joblib.dump(df_cleaned, DATA_PATH)

        model_trained = True
        print("Training successful.")
        return True, "Model trained successfully."
    except Exception as e:
        print(f"Training failed: {e}")
        return False, f"Training failed: {e}"

# Flask Routes 
@app.route('/')
def index():
    load_model()
    return render_template('index.html', model_trained=model_trained)

@app.route('/train', methods=['POST'])
def trigger_training():
    if model_trained:
        return jsonify({'status': 'info', 'message': 'Model already trained.'})
    success, msg = train_model()
    return jsonify({'status': 'success' if success else 'error', 'message': msg})

@app.route('/chat', methods=['POST'])
def chat():
    if not model_trained:
        return jsonify({'response': "Model not trained. Please train the model first."})
    try:
        user_query = request.json.get('user_message', '').strip().lower()
        if not user_query:
            return jsonify({'response': "Please enter a question."})

        # Greeting Handling (no time-based greetings) 
        greetings = {
            "hi": "Hello! How can I assist you today?",
            "hello": "Hi there! How can I help you?",
            "hey": "Hey! What can I do for you?",
        }
         #  Farewell Handling 
        farewells = {
            "bye": "Goodbye! Have a great day!",
            "goodbye": "Take care! Feel free to return if you need more help.",
            "thank you": "You're welcome! Let me know if there's anything else.",
            "thanks": "Glad I could help! Have a nice day.",
            "see you": "See you next time!",
        }
        greet_response = is_exact_match(user_query, greetings)
        if greet_response:
            return jsonify({'response': greet_response})

        farewell_response = is_exact_match(user_query, farewells)
        if farewell_response:
            return jsonify({'response': farewell_response})

        #  Semantic Search with SentenceTransformer 
        query_embedding = model.encode(user_query, convert_to_tensor=True)
        cosine_scores = util.cos_sim(query_embedding, embeddings)[0]
        best_idx = int(np.argmax(cosine_scores))
        best_score = float(cosine_scores[best_idx])

        if best_score >= 0.5:
            return jsonify({'response': answers[best_idx]})
        else:
            return jsonify({'response': "Sorry, I couldn't find a relevant answer. Please rephrase."})
    except Exception as e:
        print(f"Chat error: {e}")
        return jsonify({'response': "An error occurred. Please try again later."}), 500

# Main 
if __name__ == '__main__':
    load_model()
    app.run(debug=True)
