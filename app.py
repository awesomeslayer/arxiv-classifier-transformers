import os
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

app = FastAPI(title="ArXiv Classifier API")

app.mount("/static", StaticFiles(directory="static"), name="static")

MODEL_PATH = "./best_model"
print("Loading model into memory...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

classifier = pipeline(
    "text-classification", 
    model=model, 
    tokenizer=tokenizer, 
    device=-1,
    top_k=None 
)
print("Model loaded successfully!")

class PredictRequest(BaseModel):
    title: str
    abstract: str = ""

@app.get("/")
async def serve_frontend():
    """Отдает главную HTML страницу"""
    return FileResponse("static/index.html")

@app.post("/predict")
async def predict(req: PredictRequest):
    """Принимает текст, возвращает топ категорий (кумулятивно 95%)"""
    title = req.title.strip()
    abstract = req.abstract.strip()
    
    text_to_predict = f"{title}. {abstract}" if abstract else title
    
    if not text_to_predict:
        return {"error": "Title is required"}

    predictions = classifier(text_to_predict[:512])[0] 
    
    top_95_preds = []
    cumulative_prob = 0.0
    
    for pred in predictions:
        top_95_preds.append({
            "label": pred["label"],
            "score": round(pred["score"], 4)
        })
        cumulative_prob += pred["score"]
        if cumulative_prob >= 0.95:
            break
            
    return {"predictions": top_95_preds}