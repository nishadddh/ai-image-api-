import os
import io
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
from huggingface_hub import InferenceClient

app = Flask(__name__)
CORS(app)

HF_TOKEN = os.environ.get("HF_TOKEN")

@app.route("/generate", methods=["POST"])
def generate_image():
    data = request.get_json()
    prompt = data.get("prompt", "").strip()

    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    if not HF_TOKEN:
        return jsonify({"error": "HF_TOKEN not configured on server"}), 500

    try:
        client = InferenceClient(
            provider="wavespeed",   # ✅ works with HF token, routes through HF
            api_key=HF_TOKEN,
        )

        image = client.text_to_image(
            prompt,
            model="black-forest-labs/FLUX.1-dev",
        )

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return jsonify({"image": image_b64})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model": "black-forest-labs/FLUX.1-dev",
        "provider": "wavespeed"
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
