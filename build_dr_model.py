"""
build_dr_model.py
-----------------
Creates a pretrained DR classification model using MobileNetV2 backbone
(ImageNet pretrained) with a 5-class DR classification head.

DR Classes (APTOS 2019 convention):
  0 - No DR
  1 - Mild DR
  2 - Moderate DR
  3 - Severe DR
  4 - Proliferative DR

Input: (224, 224, 3)
Output: models/dr_model.h5
"""

import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.optimizers import Adam

print(f"[INFO] TensorFlow version: {tf.__version__}")

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_SHAPE = (224, 224, 3)
NUM_CLASSES = 5
SAVE_PATH   = os.path.join('models', 'dr_model.h5')

os.makedirs('models', exist_ok=True)

# ── Build model ───────────────────────────────────────────────────────────────
print("[INFO] Loading MobileNetV2 with ImageNet weights ...")
base = MobileNetV2(
    include_top  = False,
    weights      = 'imagenet',
    input_shape  = INPUT_SHAPE
)
base.trainable = False   # frozen pretrained backbone

print("[INFO] Adding DR classification head (5 classes) ...")
x       = base.output
x       = GlobalAveragePooling2D()(x)
x       = BatchNormalization()(x)
x       = Dense(256, activation='relu')(x)
x       = Dropout(0.4)(x)
x       = Dense(128, activation='relu')(x)
x       = Dropout(0.3)(x)
outputs = Dense(NUM_CLASSES, activation='softmax', name='dr_output')(x)

model = Model(inputs=base.input, outputs=outputs, name='DR_MobileNetV2')
model.compile(
    optimizer = Adam(learning_rate=1e-4),
    loss      = 'sparse_categorical_crossentropy',
    metrics   = ['accuracy']
)

model.summary()
print(f"\n[INFO] Input  shape : {model.input_shape}")
print(f"[INFO] Output shape : {model.output_shape}")
print(f"[INFO] Total params : {model.count_params():,}")

# ── Save ──────────────────────────────────────────────────────────────────────
print(f"\n[INFO] Saving model to '{SAVE_PATH}' ...")
try:
    model.save(SAVE_PATH)
    print(f"[OK]  Saved with model.save() → {SAVE_PATH}")
except Exception as e:
    print(f"[WARN] model.save() failed ({e}), trying tf.keras.models.save_model ...")
    tf.keras.models.save_model(model, SAVE_PATH, save_format='h5')
    print(f"[OK]  Saved with save_format='h5' → {SAVE_PATH}")

# ── Verify round-trip ─────────────────────────────────────────────────────────
print("\n[INFO] Verifying model load ...")
from tensorflow.keras.models import load_model
loaded = load_model(SAVE_PATH)

dummy  = np.zeros((1, 224, 224, 3), dtype=np.float32)
preds  = loaded.predict(dummy, verbose=0)
labels = ['No DR', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferative DR']
tclass = labels[int(np.argmax(preds))]

print(f"[OK]  Load successful!")
print(f"      Dummy prediction → '{tclass}'")
print(f"      Class probs      → {np.round(preds[0], 4)}")
print(f"\n✅  DR model ready at: {SAVE_PATH}")
