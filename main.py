from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import shutil
from utils import predict_face_shape
import os

app = FastAPI(title="Face Shape Detector API")

UPLOAD_FOLDER = "data/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.get("/")
def home():
    return {"message": "Face Shape Detector API is running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    face_shape, confidence = predict_face_shape(file_path)
    
    return JSONResponse(content={
        "face_shape": face_shape,
        "confidence": confidence
    })
