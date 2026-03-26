import numpy as np
import cv2
import os
from dr_predict import predict_dr_class
from unet_predict import predict_dr

# 1. Create a dummy test image
test_img_path = 'test_eye.jpg'
dummy_img = np.zeros((512, 512, 3), dtype=np.uint8)
# Add some "retinal" features (circles) to avoid zero-mask if possible
cv2.circle(dummy_img, (256, 256), 200, (10, 20, 150), -1) 
cv2.imwrite(test_img_path, dummy_img)

print(f"--- Running Integration Test on {test_img_path} ---")

try:
    # 2. Test Classification
    label, confidence = predict_dr_class(test_img_path)
    print(f"[Classification] Label: {label}, Confidence: {confidence:.4f}")

    # 3. Test Segmentation
    result, mask_path = predict_dr(test_img_path)
    print(f"[Segmentation] Result: {result}, Mask saved at: {mask_path}")

    if os.path.exists(mask_path):
        print("SUCCESS: Both models executed and produced outputs.")
    else:
        print("FAILURE: Mask file not found.")

except Exception as e:
    print(f"ERROR during integration: {e}")

finally:
    # Cleanup
    if os.path.exists(test_img_path):
        os.remove(test_img_path)
    # Note: mask_path might not exist yet if unet_predict failed, so we check carefully
