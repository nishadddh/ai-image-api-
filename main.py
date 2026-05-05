import os
import requests
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow requests from your InfinityFree frontend

HF_TOKEN = os.environ.get("HF_TOKEN")  # Set this in Render environment variables
MODEL_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"

@app.route("/generate", methods=["POST"])
def generate_image():
    data = request.get_json()
    prompt = data.get("prompt", "").strip()

    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    if not HF_TOKEN:
        return jsonify({"error": "HF_TOKEN not configured on server"}), 500

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "num_inference_steps": 4,   # schnell only needs 4 steps
            "width": 512,
            "height": 512
        }
    }

    try:
        response = requests.post(MODEL_URL, headers=headers, json=payload, timeout=60)

        if response.status_code == 503:
            return jsonify({"error": "Model is loading, please wait 20 seconds and retry"}), 503

        if response.status_code != 200:
            return jsonify({"error": f"HuggingFace error: {response.text}"}), response.status_code

        # Response is raw image bytes — convert to base64
        image_bytes = response.content
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        return jsonify({"image": image_b64})

    except requests.exceptions.Timeout:
        return jsonify({"error": "Request timed out. Model may be cold-starting, try again."}), 504
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "Image generation API is running"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
