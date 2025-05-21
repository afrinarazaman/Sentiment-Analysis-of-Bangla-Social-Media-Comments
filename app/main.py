from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from transformers import BertTokenizerFast
import onnxruntime as ort
import json
import os

app = FastAPI(title="Bangla Sentiment Analysis API")

MODEL_DIR = "model_parameter"

tokenizer = BertTokenizerFast.from_pretrained(os.path.join(MODEL_DIR, "tokenizer"))

ort_session = ort.InferenceSession(
    os.path.join(MODEL_DIR, "bangla_bert.onnx"),
    providers=['CPUExecutionProvider']  
)


with open(os.path.join(MODEL_DIR, "label_map.json"), "r") as f:
    label_map = json.load(f)

def preprocess(text: str):
    """Tokenize text for BERT"""
    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="np"
    )
    return {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64)
    }

def predict(text: str):
    
    inputs = preprocess(text)
    
    logits = ort_session.run(
        None, 
        {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"]
        }
    )[0]
    
    pred_id = np.argmax(logits, axis=1)[0]
    return label_map[str(pred_id)]

class TextRequest(BaseModel):
    text: str

@app.post("/predict")
async def predict_sentiment(request: TextRequest):
    try:
        prediction = predict(request.text)
        return {
            "text": request.text,
            "sentiment": prediction
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
async def home():
    return {"message": "Bangla Sentiment Analysis API - Use /predict"}