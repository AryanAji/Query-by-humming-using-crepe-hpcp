import os
import numpy as np
import librosa
import crepe

SONGS_PATH = "./songs"
EMBED_PATH = "./embeddings"
TARGET_SR = 16000

def extract_crepe_embedding(audio, sr, model_capacity='tiny'):
    # Confidence gating ensures we only average frames with enough pitch certainty
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

def process_songs():
    if not os.path.exists(EMBED_PATH):
        os.makedirs(EMBED_PATH)

    for file in os.listdir(SONGS_PATH):
        if file.endswith(".wav"):
            file_path = os.path.join(SONGS_PATH, file)
            print(f"[reference_extractor] Creating HPCP+CREPE embedding for {file}...")

            audio, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)
            c_emb = extract_crepe_embedding(audio, sr, 'tiny')
            h_emb = extract_hpcp(audio, sr)
            combined_emb = np.concatenate([c_emb, h_emb], axis=0)

            base_name = os.path.splitext(file)[0]
            out_path = os.path.join(EMBED_PATH, f"{base_name}_embed.npy")
            np.save(out_path, combined_emb)
            print(f"   Combined shape: {combined_emb.shape}")
            print(f"   Saved to {out_path}")

if __name__ == "__main__":
    process_songs()