from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import os
import numpy as np
import queue
import threading
import librosa
import crepe
import sounddevice as sd

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins='*')

EMBED_PATH = "./embeddings"
TARGET_SR = 16000
CHUNK_SECONDS = 4
CHUNK_SAMPLES = TARGET_SR * CHUNK_SECONDS

audio_queue = queue.Queue()
is_recording = False
stream = None
reference_dict = {}

def load_reference_embeddings():
    refs = {}
    for file in os.listdir(EMBED_PATH):
        if file.endswith("_embed.npy"):
            name = file.replace("_embed.npy", "")
            path = os.path.join(EMBED_PATH, file)
            emb = np.load(path)
            refs[name] = emb
    return refs

def extract_crepe_embedding(audio, sr, model_capacity='tiny'):
    time_vals, frequency, confidence, activation = crepe.predict(
        audio, sr, viterbi=True, model_capacity=model_capacity, step_size=10
    )
    valid_indices = (confidence >= 0.3)
    if np.any(valid_indices):
        emb = np.mean(activation[valid_indices], axis=0)
    else:
        emb = np.mean(activation, axis=0)
    return emb

def extract_hpcp(audio, sr):
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_chroma=12)
    return np.mean(chroma, axis=1)

def get_combined_embedding(audio, sr):
    c_emb = extract_crepe_embedding(audio, sr, 'tiny')
    h_emb = extract_hpcp(audio, sr)
    return np.concatenate([c_emb, h_emb], axis=0)

def cosine_similarity(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-9:
        return 0.0
    return float(np.dot(a, b) / denom)

def match_embedding(query_emb, references):
    results = []
    for song_name, ref_emb in references.items():
        sim = cosine_similarity(query_emb, ref_emb)
        results.append((song_name, sim))
    # Sort by descending similarity
    results.sort(key=lambda x: x[1], reverse=True)
    return results

def audio_callback(indata, frames, time_info, status):
    global is_recording
    if status:
        print(f"[audio_callback] Status: {status}")
    if is_recording:
        audio_queue.put(indata.copy())

def process_audio():
    buffer = []
    while is_recording:
        try:
            chunk = audio_queue.get(timeout=0.1)
            chunk = np.squeeze(chunk)
            buffer.extend(chunk)

            if len(buffer) >= CHUNK_SAMPLES:
                data_array = np.array(buffer[:CHUNK_SAMPLES], dtype=np.float32)
                buffer = buffer[CHUNK_SAMPLES:]

                # Resample from 44100 -> 16k for CREPE
                audio = librosa.resample(data_array, orig_sr=44100, target_sr=TARGET_SR)
                query_emb = get_combined_embedding(audio, TARGET_SR)

                matches = match_embedding(query_emb, reference_dict)
                top3 = matches[:3]

                # Send results to frontend
                out = [{"name": m[0], "similarity": m[1]} for m in top3]
                socketio.emit('update_results', {"matches": out})

        except queue.Empty:
            continue
        except Exception as e:
            print("[process_audio] Error:", e)
            continue

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('start_recording')
def handle_start_recording():
    global is_recording, stream
    if not is_recording:
        is_recording = True
        stream = sd.InputStream(channels=1, samplerate=44100, callback=audio_callback)
        stream.start()
        threading.Thread(target=process_audio, daemon=True).start()
        emit('recording_status', {'status': 'started'})

@socketio.on('stop_recording')
def handle_stop_recording():
    global is_recording, stream
    if is_recording:
        is_recording = False
        if stream:
            stream.stop()
            stream.close()
        emit('recording_status', {'status': 'stopped'})

if __name__ == '__main__':
    print("[app] Loading HPCP+CREPE references with confidence gating...")
    reference_dict = load_reference_embeddings()
    print(f"[app] Loaded {len(reference_dict)} references.")
    socketio.run(app, debug=True)
