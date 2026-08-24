import io
import os
import json
import numpy as np
import sounddevice as sd
from scipy.io import wavfile
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from deepgram import DeepgramClient
from google import genai
from record import record_audio

load_dotenv()

# Initialize API Clients
deepgram = DeepgramClient(api_key=os.getenv("DEEPGRAM_API_KEY"))
gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Load Embedding Model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# 1. Load Knowledge Base & Precompute Embeddings
with open("qna_db.json", "r", encoding="utf-8") as f:
    qna_database = json.load(f)["data"]

db_texts = [f"Q: {item['question']} A: {item['answer']}" for item in qna_database]
db_embeddings = embedder.encode(db_texts).astype("float32")

def run_audio_qna_pipeline(audio_file_path):
    # 2. Transcribe Audio (Deepgram Nova-2)
    with open(audio_file_path, "rb") as audio:
        stt_response = deepgram.listen.v1.media.transcribe_file(
            request=audio.read(),
            model="nova-2",
            smart_format=True,
        )
    user_transcript = stt_response.results.channels[0].alternatives[0].transcript
    print(f"\n[Transcribed Question]: {user_transcript}")

    # 3. Perform Vector Search (brute-force nearest neighbors via NumPy)
    query_vector = embedder.encode([user_transcript]).astype("float32")
    distances = np.linalg.norm(db_embeddings - query_vector, axis=1)
    indices = np.argsort(distances)[:2]
    retrieved_context = "\n".join([db_texts[idx] for idx in indices])
    print(f"\n[Retrieved Context]:\n{retrieved_context}")

    # 4. Generate Answer (Gemini API)
    prompt = f"""
    Answer the user's question accurately using ONLY the provided Knowledge Base context.
    
    Context:
    {retrieved_context}
    
    User Question: {user_transcript}
    """
    
    gemini_response = gemini.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )
    answer_text = gemini_response.text
    print(f"\n[Gemini Response]: {answer_text}")

    # 5. Convert Response to Audio (Deepgram Aura-2)
    audio_bytes = b"".join(
        deepgram.speak.v1.audio.generate(
            text=answer_text,
            model="aura-2-asteria-en",
            encoding="linear16",
            container="wav",
            sample_rate=24000,
        )
    )

    output_audio_path = "output_response.wav"
    with open(output_audio_path, "wb") as f:
        f.write(audio_bytes)
    print(f"\n[Output Audio Saved]: {output_audio_path}")

    # 6. Speak the response aloud
    sample_rate, audio_data = wavfile.read(io.BytesIO(audio_bytes))
    print("\n[Speaking response aloud]...")
    sd.play(audio_data, sample_rate)
    sd.wait()

if __name__ == "__main__":
    # Record 5 seconds of spoken audio from mic to user_input.wav
    record_audio(output_filename="user_input.wav", duration=5)

    # Run pipeline
    run_audio_qna_pipeline("user_input.wav")