from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import InferenceClient
import base64
import os
import io

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ NEW CLIENT (important)
client = InferenceClient(
    provider="nscale",
    api_key=os.getenv("HF_TOKEN"),
)

@app.get("/")
def home():
    return {"status": "running"}

@app.post("/generate")
def generate(data: dict):
    prompt = data.get("prompt")

    if not prompt:
        return {"error": "Prompt required"}

    try:
        # 🔥 Generate image using FLUX
        image = client.text_to_image(
            prompt,
            model="black-forest-labs/FLUX.1-schnell",
        )

        # Convert PIL image → base64
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode()

        return {"image": img_str}

    except Exception as e:
        return {"error": str(e)}
