from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests
import base64
import os

app = FastAPI()

# allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_URL = "https://api-inference.huggingface.co/models/stabilityai/sdxl-turbo"
HF_TOKEN = os.getenv("HF_TOKEN")  # secure token

@app.get("/")
def home():
    return {"status": "running"}

@app.post("/generate")
def generate(data: dict):
    prompt = data["prompt"]

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    response = requests.post(API_URL, headers=headers, json={"inputs": prompt})

    image_base64 = base64.b64encode(response.content).decode("utf-8")

    return {"image": image_base64}
