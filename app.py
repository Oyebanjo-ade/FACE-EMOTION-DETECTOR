# app.py
# Flask app to detect emotions from uploaded images or webcam stream.

from flask import Flask, render_template, request, redirect, url_for
import sqlite3, os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Load model and other dependencies
model = load_model("face_emotionModel.h5")
emotion_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Initialize database
def init_db():
    with sqlite3.connect("database.db") as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            emotion TEXT,
            image_path TEXT
        )
        """)
init_db()

# Routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    name = request.form.get("name", "")
    file = request.files.get("image")

    if not file or file.filename == "":
        return "No file uploaded!", 400

    # Secure filename and save original file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    # --- NEW IMAGE HANDLING ---
    # Read uploaded file in memory and decode with OpenCV
    in_memory_file = file.read()
    img = cv2.imdecode(np.frombuffer(in_memory_file, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return "Failed to read image. Make sure it's a valid image file.", 400
    # ---------------------------

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    emotion_result = "No face detected."

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = np.expand_dims(roi, axis=(0, -1))
        preds = model.predict(roi)
        label = emotion_labels[np.argmax(preds)]
        emotion_result = f"You look {label}. How are you feeling?"

        # Draw on image
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Save processed image
    cv2.imwrite(filepath, img)

    # Save data to DB
    with sqlite3.connect("database.db") as conn:
        conn.execute(
            "INSERT INTO users (name, emotion, image_path) VALUES (?, ?, ?)",
            (name, emotion_result, filepath)
        )

    return render_template("result.html", name=name, emotion=emotion_result, image_path=filepath)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
