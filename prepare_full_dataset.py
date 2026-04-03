import os
import shutil
import pandas as pd

csv_path = "train.csv"
images_path = "train_images"
output_path = "dataset/train"

df = pd.read_csv(csv_path)

for index, row in df.iterrows():
    img_name = row['id_code'] + ".png"
    label = str(row['diagnosis'])

    src = os.path.join(images_path, img_name)
    dst = os.path.join(output_path, label, img_name)

    if os.path.exists(src):
        shutil.copy(src, dst)

print("🔥 FULL DATASET SORTED")