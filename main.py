import os
import base64
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

HF_TOKEN = os.environ.get("HF_TOKEN")

# ✅ Free HF Inference API — no provider, no Replicate, no billing
API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"

@app.route("/generate", methods=["POST"])
def generate_image():
    data = request.get_json()
    prompt = data.get("prompt", "").strip()

    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    if not HF_TOKEN:
        return jsonify({"error": "HF_TOKEN not configured on server"}), 500

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}"
    }

    payload = {
        "inputs": prompt,
        "parameters": {
            "num_inference_steps": 25,
            "width": 512,
            "height": 512
        }
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=120)

        # Model is cold starting — tell frontend to retry
        if response.status_code == 503:
            return jsonify({"error": "Model is warming up, please wait 20 seconds and try again"}), 503

        if response.status_code != 200:
            return jsonify({"error": f"HF error {response.status_code}: {response.text}"}), response.status_code

        # HF returns raw image bytes
        image_b64 = base64.b64encode(response.content).decode("utf-8")
        return jsonify({"image": image_b64})

    except requests.exceptions.Timeout:
        return jsonify({"error": "Timed out, please try again"}), 504
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model": "runwayml/stable-diffusion-v1-5",
        "provider": "huggingface-free"
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
