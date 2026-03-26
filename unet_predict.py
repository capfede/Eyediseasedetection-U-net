import numpy as np
import cv2
import os
from model_unet import unet   # IMPORTANT: from your copied file

# load model
model = unet()

# TEMP: comment this for now (we don’t have trained weights yet)
# model.load_weights("unet_model.h5")


def predict_dr(image_path):
    # read image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (256, 256))

    # normalize
    img = img / 255.0

    # reshape
    img = np.reshape(img, (1, 256, 256, 3))

    # predict
    mask = model.predict(img)[0]

    # convert mask to binary image
    mask = (mask > 0.5).astype(np.uint8) * 255

    # save mask
    mask_filename = "mask_" + os.path.basename(image_path)
    mask_path = os.path.join("static/masks", mask_filename)

    cv2.imwrite(mask_path, mask)

    # simple detection logic
    if np.sum(mask) > 1000:
        result = "Diabetic Retinopathy Detected"
    else:
        result = "Normal"

    return result, mask_path