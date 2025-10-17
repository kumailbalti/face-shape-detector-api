import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load your trained model (example)
model = load_model('models/face_shape_model.h5')

def preprocess_image(image_path):
    """Load and preprocess image for prediction"""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_face_shape(image_path):
    """Predict the face shape"""
    img = preprocess_image(image_path)
    preds = model.predict(img)
    shapes = ['Oval', 'Round', 'Square', 'Heart', 'Diamond']
    idx = np.argmax(preds)
    return shapes[idx], float(preds[0][idx])
