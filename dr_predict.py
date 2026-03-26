import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

model = load_model("models/dr_model.h5")

classes = ["Normal", "Mild", "Moderate", "Severe", "Proliferative DR"]

def predict_dr_class(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))

    img = preprocess_input(img)   # 🔥 FIXED

    img = np.reshape(img, (1, 224, 224, 3))

    prediction = model.predict(img)

    class_index = np.argmax(prediction)
    confidence = float(np.max(prediction))

    result = classes[class_index]

    print("RAW PREDICTION:", prediction)  # DEBUG

    return result, confidence