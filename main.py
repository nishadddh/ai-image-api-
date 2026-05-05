import os
import io
import base64
import requests
import time
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

REPLICATE_TOKEN = os.environ.get("REPLICATE_TOKEN")

@app.route("/generate", methods=["POST"])
def generate_image():
    data = request.get_json()
    prompt = data.get("prompt", "").strip()

    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    if not REPLICATE_TOKEN:
        return jsonify({"error": "REPLICATE_TOKEN not configured on server"}), 500

    try:
        headers = {
            "Authorization": f"Bearer {REPLICATE_TOKEN}",
            "Content-Type": "application/json",
            "Prefer": "wait"
        }

        payload = {
            "version": "39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
            "input": {
                "prompt": prompt,
                "num_inference_steps": 25,
                "width": 1024,
                "height": 1024
            }
        }

        response = requests.post(
            "https://api.replicate.com/v1/predictions",
            headers=headers,
            json=payload,
            timeout=120
        )

        if response.status_code not in [200, 201]:
            return jsonify({"error": f"Replicate error: {response.text}"}), response.status_code

        result = response.json()

        # Poll until done
        while result.get("status") not in ["succeeded", "failed", "canceled"]:
            time.sleep(2)
            poll_resp = requests.get(result["urls"]["get"], headers=headers, timeout=60)
            result = poll_resp.json()

        if result.get("status") != "succeeded":
            return jsonify({"error": "Generation failed: " + str(result.get("error"))}), 500

        # Download image and convert to base64
        image_url = result["output"][0]
        img_response = requests.get(image_url, timeout=60)
        image_b64 = base64.b64encode(img_response.content).decode("utf-8")

        return jsonify({"image": image_b64})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model": "stability-ai/sdxl",
        "provider": "replicate"
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
