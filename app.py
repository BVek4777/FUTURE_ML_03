import os
import json
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from load_data import data_loader  # your module that returns raw DataFrame with 'qa' column

# --- Configuration ---
MODEL_DIR = 'model'
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'vectorizer.joblib')
TFIDF_MATRIX_PATH = os.path.join(MODEL_DIR, 'tfidf_matrix.joblib')
DATA_PATH = os.path.join(MODEL_DIR, 'data.joblib')

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Global Variables ---
vectorizer = None
tfidf_matrix = None
df = None  # will hold DataFrame with columns ['question','answer']
model_trained = False

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Load Model from disk ---
def load_model():
    global vectorizer, tfidf_matrix, df, model_trained
    try:
        vectorizer = joblib.load(VECTORIZER_PATH)
        tfidf_matrix = joblib.load(TFIDF_MATRIX_PATH)
        df = joblib.load(DATA_PATH)
        model_trained = True
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Model load failed: {e}")
        model_trained = False

# --- Train TF-IDF model ---
def train_model():
    global vectorizer, tfidf_matrix, df, model_trained
    try:
        # Load raw data
        df_raw = data_loader()
        # Parse 'qa' JSON strings or dicts
        records = []
        for entry in df_raw['qa']:
            if isinstance(entry, str):
                try:
                    entry = json.loads(entry)
                except json.JSONDecodeError:
                    continue
            # entry should be a dict with key 'knowledge'
            for item in entry.get('knowledge', []):
                q = item.get('customer_summary_question', '').strip()
                a = item.get('agent_summary_solution', '').strip()
                if q:
                    records.append({'question': q, 'answer': a or 'No solution provided.'})
        # Build DataFrame
        if not records:
            return False, "Training failed: No valid QA pairs found in 'qa' column."
        df_cleaned = pd.DataFrame(records)
        # Vectorize questions
        vectorizer = TfidfVectorizer(stop_words='english', max_df=0.9, min_df=2)
        tfidf_matrix = vectorizer.fit_transform(df_cleaned['question'])
        # Save to disk
        joblib.dump(vectorizer, VECTORIZER_PATH)
        joblib.dump(tfidf_matrix, TFIDF_MATRIX_PATH)
        joblib.dump(df_cleaned, DATA_PATH)
        # Update globals
        df = df_cleaned
        model_trained = True
        print("Training successful.")
        return True, "Model trained successfully."
    except Exception as e:
        print(f"Training failed: {e}")
        return False, f"Training failed: {e}"

# --- Flask Routes ---
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
        user_query = request.json.get('user_message', '').strip()
        if not user_query:
            return jsonify({'response': "Please enter a question."})
        # Vectorize user query
        query_vec = vectorizer.transform([user_query])
        sims = cosine_similarity(query_vec, tfidf_matrix).flatten()
        best_idx = np.argmax(sims)
        best_score = sims[best_idx]
        # Threshold check
        if best_score >= 0.1:
            answer = df.iloc[best_idx]['answer']
            return jsonify({'response': answer})
        else:
            return jsonify({'response': "Sorry, I couldn't find a relevant answer. Please rephrase."})
    except Exception as e:
        print(f"Chat error: {e}")
        return jsonify({'response': "An error occurred. Please try again later."}), 500

# --- Main ---
if __name__ == '__main__':
    load_model()
    app.run(debug=True)
