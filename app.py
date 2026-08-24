import os
import json
import faiss
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from deepgram import DeepgramClient
from google import genai
from record import record_audio

load_dotenv()

# Initialize API Clients
deepgram = DeepgramClient(os.getenv("DEEPGRAM_API_KEY"))
gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Load Embedding Model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# 1. Load Knowledge Base & Build FAISS Index
with open("qna_db.json", "r") as f:
    qna_database = json.load(f)

db_texts = [f"Q: {item['question']} A: {item['answer']}" for item in qna_database]
db_embeddings = embedder.encode(db_texts).astype("float32")

dimension = db_embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(db_embeddings)

def run_audio_qna_pipeline(audio_file_path):
    # 2. Transcribe Audio (Deepgram Nova-2)
    with open(audio_file_path, "rb") as audio:
        stt_response = deepgram.listen.rest.v("1").transcribe_file(
            {"buffer": audio},
            {"model": "nova-2", "smart_format": True}
        )
    user_transcript = stt_response.results.channels[0].alternatives[0].transcript
    print(f"\n[Transcribed Question]: {user_transcript}")

    # 3. Perform Vector Search (FAISS)
    query_vector = embedder.encode([user_transcript]).astype("float32")
    distances, indices = faiss_index.search(query_vector, k=2)
    retrieved_context = "\n".join([db_texts[idx] for idx in indices[0]])
    print(f"\n[Retrieved Context]:\n{retrieved_context}")

    # 4. Generate Answer (Gemini API)
    prompt = f"""
    Answer the user's question accurately using ONLY the provided Knowledge Base context.
    
    Context:
    {retrieved_context}
    
    User Question: {user_transcript}
    """
    
    gemini_response = gemini.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    answer_text = gemini_response.text
    print(f"\n[Gemini Response]: {answer_text}")

    # 5. Convert Response to Audio (Deepgram Aura-2)
    tts_response = deepgram.speak.v("1").generate(
        text=answer_text,
        model="aura-2-asteria-en"
    )
    
    output_audio_path = "output_response.mp3"
    with open(output_audio_path, "wb") as f:
        f.write(tts_response.stream.read())
        
    print(f"\n[Output Audio Saved]: {output_audio_path}")

if __name__ == "__main__":
    # Record 5 seconds of spoken audio from mic to user_input.wav
    record_audio(output_filename="user_input.wav", duration=5)

    # Run pipeline
    run_audio_qna_pipeline("user_input.wav")