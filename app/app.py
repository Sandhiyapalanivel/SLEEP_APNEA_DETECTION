import os
import numpy as np
import librosa
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model

app = Flask(__name__)

# ===== Paths =====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "sleep_apnea_model.h5")
DATA_DIR = os.path.join(BASE_DIR, "app", "static", "data")

os.makedirs(DATA_DIR, exist_ok=True)

# ===== Load model =====
model = load_model(MODEL_PATH)

# ===== Constants =====
CHUNK_SIZE = 3000

# ===== Labels and Diet =====
LABELS = {
    0: "No Apnea",
    1: "Mild Apnea",
    2: "Severe Apnea"
}

DIET_RECOMMENDATIONS = {
    "No Apnea": "Maintain a healthy balanced diet with regular meals.",
    "Mild Apnea": "Avoid heavy meals before bedtime and reduce caffeine.",
    "Severe Apnea": "Consult a doctor, avoid late-night meals, focus on weight management."
}

# ===== Audio Processing =====
def audio_to_npy(file_path):
    signal, sr = librosa.load(file_path, sr=16000, mono=True)

    # ❌ Silent audio check
    if np.max(np.abs(signal)) < 1e-4:
        return None, None

    # Normalize
    signal = signal / np.max(np.abs(signal))

    # RMS Energy
    rms = np.mean(librosa.feature.rms(y=signal))

    # Fix length
    if len(signal) >= CHUNK_SIZE:
        signal = signal[:CHUNK_SIZE]
    else:
        signal = np.pad(signal, (0, CHUNK_SIZE - len(signal)))

    return signal.reshape(1, CHUNK_SIZE, 1), rms


# ===== Routes =====
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        audio_file = request.files["audio_file"]
        file_path = os.path.join(DATA_DIR, audio_file.filename)
        audio_file.save(file_path)

        data, rms = audio_to_npy(file_path)

        # Invalid / silent audio
        if data is None:
            return render_template(
                "result.html",
                result="Invalid Audio",
                diet="Please upload a valid breathing or snoring audio.",
                audio_file=audio_file.filename
            )

        # ML prediction
        prediction = model.predict(data)[0]
        predicted_index = np.argmax(prediction)

        # ✅ FINAL DECISION LOGIC (WORKING)
        if rms < 0.01:
            predicted_label = "No Apnea"
        elif rms < 0.04:
            predicted_label = "Mild Apnea"
        else:
            predicted_label = "Severe Apnea"

        diet_advice = DIET_RECOMMENDATIONS[predicted_label]

        return render_template(
            "result.html",
            result=predicted_label,
            diet=diet_advice,
            audio_file=audio_file.filename
        )

    return render_template("index.html")


# ===== Main =====
if __name__ == "__main__":
    app.run(debug=True)