from flask import Flask, request, jsonify
from ultralytics import YOLO
import base64
import numpy as np
import cv2

app = Flask(__name__)

# 🔥 Cargar modelo UNA sola vez
model = YOLO("yolov8n.pt")

@app.route("/")
def home():
    return "Servidor de detección activo"

@app.route("/detect", methods=["POST"])
def detect():
    try:
        data = request.get_json()

        if not data or "image" not in data:
            return jsonify({"error": "No image sent"}), 400

        image_base64 = data["image"]

        # 🔥 limpiar saltos y padding
        image_base64 = image_base64.replace("\n", "").replace("\r", "")
        while len(image_base64) % 4 != 0:
            image_base64 += "="

        # Decodificar base64
        image_bytes = base64.b64decode(image_base64)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Image decode failed"}), 400

        # 🔥 CLAVE: reducir tamaño de inferencia
        results = model(img, imgsz=224, device="cpu")

        persons = 0
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == 0:  # clase 0 = persona
                    persons += 1

        return jsonify({"persons_detected": persons})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
