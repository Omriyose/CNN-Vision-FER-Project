from fastapi import FastAPI
from tensorflow.keras.models import load_model
import io
import numpy as np
from fastapi import File, UploadFile
from PIL import Image
from fastapi.responses import FileResponse
from PIL import ImageOps

app = FastAPI()
model = load_model('fer_cnn_model_v4.keras')
CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

@app.get("/")
def home():
    return FileResponse("index.html")

@app.get("/classes")
def classes():
    return CLASS_NAMES

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img)
    img = ImageOps.fit(img.convert('L'), (48, 48))
    x = np.array(img) / 255
    x = x.reshape(1, 48, 48, 1)
    probs = model.predict(x, verbose=0)[0]
    i = int(np.argmax(probs))
    all_probs = {name: round(float(p) * 100, 1) for name, p in zip(CLASS_NAMES, probs)}
    if float(probs[i]) < 0.6:
        return {"emotion": "unknown", "confidence": round(float(probs[i]) * 100, 1), "all": all_probs}
    return {"emotion": CLASS_NAMES[i], "confidence": round(float(probs[i]) * 100, 1), "all": all_probs}