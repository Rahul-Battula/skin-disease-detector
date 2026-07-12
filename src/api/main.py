import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from torchvision import transforms
from PIL import Image
import io

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile
from utils.llm_advisor import get_advice, llm
from fastapi.responses import JSONResponse

from models.model import create_model

app = FastAPI(title="Skin Disease Detector API")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://skin-disease-detector-virid.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Config ----
CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model.pth')
DEVICE = torch.device('cpu')  # inference on CPU is fine, no training happening here

# ---- Load model once at startup ----
model = create_model(num_classes=len(CLASS_NAMES), pretrained=False)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ---- Preprocessing (must match validation transform used in training) ----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.get("/")
def root():
    return {"message": "Skin Disease Detector API is running"}

CONFIDENCE_THRESHOLD = 0.6  # be conservative before committing to a prediction
MARGIN_THRESHOLD = 0.15     # if top 2 predictions are this close, model is "unsure"

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        sorted_probs, sorted_idx = torch.sort(probabilities, descending=True)

    top_confidence = sorted_probs[0].item()
    second_confidence = sorted_probs[1].item()
    margin = top_confidence - second_confidence
    predicted_class = CLASS_NAMES[sorted_idx[0].item()]
    confidence_score = round(top_confidence, 3)

    is_uncertain = top_confidence < CONFIDENCE_THRESHOLD or margin < MARGIN_THRESHOLD

    if is_uncertain:
        advice = get_advice(predicted_class=None, confidence=confidence_score, no_disease=True)
        return {
            "result": "No disease found",
            "confidence": confidence_score,
            "advice": advice
        }

    advice = get_advice(predicted_class=predicted_class, confidence=confidence_score)
    return {
        "result": predicted_class,
        "confidence": confidence_score,
        "advice": advice,
        "all_probabilities": {
            CLASS_NAMES[i]: round(probabilities[i].item(), 3) for i in range(len(CLASS_NAMES))
        }
    }


# ---- Chatbot ----
class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []

CHAT_SYSTEM_PROMPT = """You are a friendly skin care assistant for DermEx.

Answer questions about skin care: routines, ingredients, common conditions, sun protection, and general dermatology knowledge.

STRICT RULES:
- Keep replies SHORT: 2-3 sentences maximum, unless the user explicitly asks for more detail or a list.
- Do not repeat the disclaimer ("not a substitute for medical advice") in every message — only mention it once if the user describes a specific personal symptom or asks for a diagnosis.
- Do not pad responses with generic openers like "It's nice to meet you" or "I'm here to help with any questions" — just answer directly.
- Do NOT diagnose any specific condition the user describes about themselves. Suggest DermEx's image analysis feature or a dermatologist instead.
- If asked about anything unrelated to skin/hair/general health, briefly redirect to skin topics.
- Be warm but concise, like a quick helpful text message, not an essay."""

@app.post("/chat")
async def chat(request: ChatRequest):
    messages = [SystemMessage(content=CHAT_SYSTEM_PROMPT)]

    for turn in request.history[-6:]:  # keep last 6 messages for context, avoid huge prompts
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        else:
            messages.append(SystemMessage(content=turn["content"]))

    messages.append(HumanMessage(content=request.message))

    response = llm.invoke(messages)
    return {"reply": response.content}