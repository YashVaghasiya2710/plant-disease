from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

print(tf.__version__)
app = FastAPI()

# Load the SavedModel as a TFSMLayer
MODEL = tf.saved_model.load("C:/Users/Yash/OneDrive/Desktop/code/plant-disease/models/6/saved_model", tags=None, options=None)
class_names = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy']


@app.get("/ping")
async def ping():
    return "HELLO I AM ALIVE"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

def preprocess_image(image):
    # Resize image to (256, 256) and normalize pixel values
    image = tf.image.resize(image, (256, 256))
    image /= 255.0  # Normalize pixel values to [0, 1]
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    # Read uploaded image
    image = read_file_as_image(await file.read())
    
    # Preprocess the image
    image = preprocess_image(image)
    
    # Add batch dimension
    image = tf.expand_dims(image, 0)
    
    # Make prediction
    prediction = MODEL.predict(image)
    predicted_class = class_names[np.argmax(prediction[0])]
    confidence = float(np.max(prediction[0]))
    
    return {
        'class': predicted_class,
        'confidence': confidence
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)


