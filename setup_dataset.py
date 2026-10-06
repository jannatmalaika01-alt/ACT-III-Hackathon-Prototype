import os
import csv
from PIL import Image, ImageDraw
import random

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TEST_DIR = os.path.join(DATA_DIR, "test")
CSV_PATH = os.path.join(DATA_DIR, "test_cases.csv")

os.makedirs(TEST_DIR, exist_ok=True)

# Define our 10 test samples and their ground truth labels
SAMPLES = [
    ("test_01.png", "normal"),
    ("test_02.png", "bearing_wear"),
    ("test_03.png", "bearing_wear"),
    ("test_04.png", "structural_crack"),
    ("test_05.png", "corrosion"),
    ("test_06.png", "corrosion"),
    ("test_07.png", "bearing_wear"),
    ("test_08.png", "structural_crack"),
    ("test_09.png", "normal"),
    ("test_10.png", "normal"),
]

def generate_synthetic_image(filename, label, path):
    img = Image.new("RGB", (224, 224), color=(200, 200, 200))
    draw = ImageDraw.Draw(img)

    if label == "bearing_wear":
        # Draw circular wear patterns
        for r in range(20, 90, 15):
            draw.ellipse([112 - r, 112 - r, 112 + r, 112 + r], outline=(80, 80, 80), width=3)
    elif label == "structural_crack":
        # Draw jagged line representing a crack
        points = [(20, 30), (70, 80), (110, 90), (150, 160), (200, 190)]
        draw.line(points, fill=(20, 20, 20), width=4)
    elif label == "corrosion":
        # Draw speckles simulating rust/corrosion
        random.seed(42)
        for _ in range(300):
            x = random.randint(0, 223)
            y = random.randint(0, 223)
            draw.rectangle([x, y, x+2, y+2], fill=(150, 75, 0))

    img.save(path)

def setup():
    print("Generating local industrial test images...")
    csv_rows = [["image", "expected_label"]]

    for filename, label in SAMPLES:
        file_path = os.path.join(TEST_DIR, filename)
        generate_synthetic_image(filename, label, file_path)
        csv_rows.append([filename, label])
        print(f" Created {filename} -> Label: {label}")

    with open(CSV_PATH, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)

    print(f"\nDataset setup complete!")
    print(f"Images location: {TEST_DIR}")
    print(f"Ground truth CSV: {CSV_PATH}")

if __name__ == "__main__":
    setup()