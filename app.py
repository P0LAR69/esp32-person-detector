from flask import Flask, request, jsonify
from ultralytics import YOLO
import base64
import numpy as np
import cv2

app = Flask(__name__)
model = YOLO("yolov8n.pt")

@app.route("/")
def home():
    return "Servidor de detección activo"

@app.route("/detect", methods=["POST"])
def detect():
    data = request.json
    image_base64 = data["image"]

    # Decodificar base64
    image_bytes = base64.b64decode(image_base64)
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    results = model(img)

    persons = 0
    for r in results:
        for box in r.boxes:
            if int(box.cls) == 0:  # clase 0 = persona
                persons += 1

    return jsonify({"persons_detected": persons})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
