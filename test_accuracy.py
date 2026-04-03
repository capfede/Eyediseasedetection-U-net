import os

print("STARTING SCRIPT")

DATASET_PATH = "dataset/train"

if not os.path.exists(DATASET_PATH):
    print("Dataset not found")
    exit()

print("Folders:", os.listdir(DATASET_PATH))

def run_test():
    from dr_predict import predict_dr_class

    label_map = {
        "0": "Normal",
        "1": "Mild",
        "2": "Moderate",
        "3": "Severe",
        "4": "Proliferative DR"
    }

    correct = 0
    total = 0

    for folder in os.listdir(DATASET_PATH):
        if folder not in label_map:
            continue

        folder_path = os.path.join(DATASET_PATH, folder)

        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)

            try:
                predicted, _ = predict_dr_class(img_path)
                print(f"{img_name} -> {predicted}")

                if predicted.lower() == label_map[folder].lower():
                    correct += 1

                total += 1

            except Exception as e:
                print("Error:", e)

    if total > 0:
        accuracy = (correct / total) * 100
        print("\n🔥 FINAL ACCURACY:", accuracy)
    else:
        print("\n❌ No images processed")

run_test()