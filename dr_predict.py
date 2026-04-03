import numpy as np
import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import DepthwiseConv2D

# Fix for Keras 3 deserialization error: 'groups' is not a valid argument for DepthwiseConv2D
class FixedDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, **kwargs):
        if 'groups' in kwargs:
            kwargs.pop('groups')
        super().__init__(**kwargs)

model = None

def load_dr_model():
    global model
    if model is None:
        from tensorflow.keras.models import load_model
        print("🔥 Loading trained DR model...")
        model = load_model("models/dr_model_trained.h5", custom_objects={'DepthwiseConv2D': FixedDepthwiseConv2D})
    return model

classes = ["Normal", "Mild", "Moderate", "Severe", "Proliferative DR"]

def predict_dr_class(image_path):
    model_loaded = load_dr_model()

    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))

    img = preprocess_input(img)

    img = np.reshape(img, (1, 224, 224, 3))

    prediction = model_loaded.predict(img)

    class_index = np.argmax(prediction)
    confidence = float(np.max(prediction)) * 100

    result = classes[class_index]

    return result, confidence