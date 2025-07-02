from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil
import uuid
import os
import cv2
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import torch
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

from app.parser import parse_document
from app.saver import save_to_json

# FastAPI App Initialization
app = FastAPI()

# CORS for Pneumonia API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

templates = Jinja2Templates(directory="app/templates")

# Sentiment Analysis Model Load

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()
sentiment_model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)

# News Categorization Model Load

news_model = joblib.load('models/nb_model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')


# Pneumonia Detection Model Load
pneumonia_model = load_model('models/pneumonia_model.h5')

# Common Folders
UPLOAD_FOLDER = "uploads"
DATA_FOLDER = "data"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)

# Routes

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# Cartoonify

@app.get("/cartoonify", response_class=HTMLResponse)
async def cartoonify_page(request: Request):
    return templates.TemplateResponse("cartoonify.html", {"request": request})

@app.post("/cartoonify")
async def cartoonify(request: Request, file: UploadFile = File(...)):
    filename = file.filename
    upload_path = f"static/uploads/{filename}"
    cartoon_path = f"static/uploads/cartoon_{filename}"

    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    img = cv2.imread(upload_path)

    def edge_mask(img, line_size=7, blur_value=7):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.medianBlur(gray, blur_value)
        edges = cv2.adaptiveThreshold(gray_blur, 255,
                                      cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY,
                                      blockSize=line_size,
                                      C=blur_value)
        return edges

    def color_quantization(img, k=9):
        data = np.float32(img).reshape((-1, 3))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
        _, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        result = center[label.flatten()]
        return result.reshape(img.shape)

    edges = edge_mask(img)
    quantized = color_quantization(img)
    blurred = cv2.bilateralFilter(quantized, d=7, sigmaColor=200, sigmaSpace=200)
    cartoon = cv2.bitwise_and(blurred, blurred, mask=edges)

    cv2.imwrite(cartoon_path, cartoon)

    return templates.TemplateResponse("cartoonify.html", {
        "request": request,
        "original_img": f"/{upload_path}",
        "result_img": f"/{cartoon_path}"
    })


# Customer Sentiment

@app.get("/sentiment", response_class=HTMLResponse)
async def sentiment_page(request: Request):
    return templates.TemplateResponse("sentiment.html", {
        "request": request,
        "input_text": "",
        "vader": None,
        "roberta": None
    })

@app.post("/sentiment", response_class=HTMLResponse)
async def analyze_sentiment(request: Request, text: str = Form(...)):
    vader_scores = sia.polarity_scores(text)
    encoded_text = tokenizer(text, return_tensors='pt', truncation=True)
    with torch.no_grad():
        output = sentiment_model(**encoded_text)
    scores = softmax(output.logits[0].numpy())
    roberta_scores = {"Negative": float(scores[0]), "Neutral": float(scores[1]), "Positive": float(scores[2])}

    return templates.TemplateResponse("sentiment.html", {
        "request": request,
        "input_text": text,
        "vader": vader_scores,
        "roberta": roberta_scores
    })


# News Categorization

@app.get("/news", response_class=HTMLResponse)
async def news_page(request: Request):
    return templates.TemplateResponse("news.html", {"request": request, "prediction": None})

@app.post("/news", response_class=HTMLResponse)
async def predict_news(request: Request, news: str = Form(...)):
    text_vec = vectorizer.transform([news])
    prediction = news_model.predict(text_vec)[0]
    return templates.TemplateResponse("news.html", {"request": request, "prediction": prediction})


# OCR Parser

@app.get("/ocr", response_class=HTMLResponse)
async def ocr_page(request: Request):
    return templates.TemplateResponse("ocr.html", {"request": request})

@app.post("/ocr", response_class=HTMLResponse)
async def upload(request: Request, file: UploadFile = File(...)):
    file_id = uuid.uuid4().hex
    file_path = f"{UPLOAD_FOLDER}/{file_id}_{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    parsed_data = parse_document(file_path)
    json_path = f"{DATA_FOLDER}/{file_id}.json"
    save_to_json(parsed_data, json_path)

    return templates.TemplateResponse("ocr.html", {
        "request": request,
        "filename": file.filename,
        "image_url": "/" + file_path,
        "data": parsed_data,
        "json_path": "/" + json_path
    })

@app.get("/download/{filename}")
async def download(filename: str):
    return FileResponse(f"data/{filename}", media_type='application/json', filename=filename)


# Pneumonia Detection

@app.get("/pneumonia", response_class=HTMLResponse)
async def pneumonia_page(request: Request):
    return templates.TemplateResponse("pneumonia.html", {"request": request, "result": None})

@app.post("/pneumonia", response_class=HTMLResponse)
async def predict_pneumonia(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert('RGB')
    img = img.resize((224, 224))

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = pneumonia_model.predict(img_array)
    result = np.argmax(prediction)

    if result == 0:
        result_text = "Normal"
    else:
        result_text = "Pneumonia"

    return templates.TemplateResponse("pneumonia.html", {
        "request": request,
        "result": result_text
    })
