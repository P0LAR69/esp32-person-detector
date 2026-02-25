from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import os

app = Flask(__name__)

# Cargar modelo nano (ligero)
model = YOLO("yolov8n.pt")

@app.route("/")
def home():
    return "Servidor de detección activo"

@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "No image sent"}), 400

    file = request.files["image"]
    img = Image.open(file.stream)

    # Redimensionar para hacerlo más ligero
    img = img.resize((320, 320))

    results = model(img)

    persons = 0

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            if label == "person":
                persons += 1

    return jsonify({"persons_detected": persons})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
