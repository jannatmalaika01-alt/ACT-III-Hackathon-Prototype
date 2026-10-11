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

HELD_OUT_LABELS = ["normal", "bearing_wear", "structural_crack", "corrosion", "normal",
                   "bearing_wear", "structural_crack", "corrosion", "normal", "corrosion"]

REPORT_DIR = os.path.join(DATA_DIR, "test_reports")

REPORTS = {  # file -> (input_type, expected_label, content)
    "s01_motor_vibration.csv": ("sensor", "bearing_wear",
        "timestamp,vibration_mm_s,bearing_temp_c\n08:00,6.9,88\n09:00,8.2,97\n10:00,8.6,101\n"),
    "s02_pipe_wall.json": ("sensor", "corrosion",
        '{"line": "B7", "nominal_wall_mm": 10, "wall_thickness_mm": 7.4}'),
    "s03_weld_crack_gauge.txt": ("sensor", "structural_crack",
        "sensor: crack gauge W3\ncrack_width_mm: 1.3\n"),
    "s04_healthy_pump.csv": ("sensor", "normal",
        "timestamp,vibration_mm_s,bearing_temp_c\n08:00,2.1,58\n09:00,2.3,61\n"),
    "d01_shift_report.txt": ("document", "bearing_wear",
        "Shift report line 2. Operator heard metallic noise from the motor and felt excessive vibration at the bearing housing. "
        "Housing was hot to the touch. Grease appeared dry around the seal."),
    "d02_pipe_inspection.txt": ("document", "corrosion",
        "Pipe inspection B7. Brown oxide flakes and surface pitting found on the lower run. Paint has blistered and rust is "
        "spreading along the weld heat-affected zone. Pitting depth approx 1.5 mm."),
    "d03_clean_audit.txt": ("document", "normal",
        "Monthly audit of frame F2. No visible cracks, no sign of rust, joints tight. The structure is free of damage and all "
        "readings are within limits."),
    "d04_weld_report.pdf": ("document", "structural_crack",
        "Weld inspection report. A crack was found along the support seam. Linear fracture propagation of 40 mm measured. "
        "Fatigue cracking is suspected near the toe of the weld."),
    "d05_scanned_crack_report.pdf": ("scan", "structural_crack",
        "SCANNED REPORT\nA crack was found on the welded support seam.\nFracture is propagating along the weld.\nHalt loads."),
}


def setup_reports():
    """Sensor + document fixtures for the non-image input types. d05 is an image-only (scanned) PDF to exercise OCR."""
    from PIL import ImageFont
    os.makedirs(REPORT_DIR, exist_ok=True)
    rows = [["file", "input_type", "expected_label"]]
    for name, (kind, label, content) in REPORTS.items():
        path = os.path.join(REPORT_DIR, name)
        if kind == "scan":
            img = Image.new("RGB", (1240, 700), "white")
            d = ImageDraw.Draw(img)
            font = ImageFont.load_default(size=36)
            for i, line in enumerate(content.split("\n")):
                d.text((60, 60 + i * 70), line, fill="black", font=font)
            img.save(path, "PDF", resolution=150)
            rows.append([name, "document", label])
        elif name.endswith(".pdf"):
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas
            c = canvas.Canvas(path, pagesize=letter)
            y = 720
            for part in content.split(". "):
                c.drawString(50, y, part.strip() + ".")
                y -= 20
            c.save()
            rows.append([name, "document", label])
        else:
            with open(path, "w", newline="") as f:
                f.write(content)
            rows.append([name, kind, label])
    with open(os.path.join(REPORT_DIR, "manifest.csv"), "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"Created {len(REPORTS)} sensor/document test reports in {REPORT_DIR}")


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

    # Held-out harder images (textured backgrounds, noise, blur). Their seeds are never used for training.
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from vision import synthetic
    for n, label in enumerate(HELD_OUT_LABELS, start=11):
        filename = f"test_{n:02d}.png"
        synthetic.generate(label, 20_000_000 + n, "hard").save(os.path.join(TEST_DIR, filename))
        csv_rows.append([filename, label])
        print(f" Created {filename} -> Label: {label} (held-out, hard)")

    with open(CSV_PATH, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)

    print(f"\nDataset setup complete!")
    print(f"Images location: {TEST_DIR}")
    print(f"Ground truth CSV: {CSV_PATH}")

if __name__ == "__main__":
    setup()
    setup_reports()