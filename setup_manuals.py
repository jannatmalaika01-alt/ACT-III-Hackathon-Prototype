import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")
os.makedirs(RAW_DATA_DIR, exist_ok=True)

MANUALS = {
    "bearing_maintenance_manual.pdf": [
        ("Page 1: General Bearing Overview", 
         "Standard operating procedures for industrial machine bearings. "
         "Regular inspection prevents catastrophic motor and axle failure. "
         "Check for unusual noise, heat accumulation, and excessive vibration during daily rounds."),
        
        ("Page 2: Bearing Wear Diagnosis & Action", 
         "Defect Type: Bearing Wear and Fatigue Scoring.\n"
         "Symptom: Circular scoring marks, metallic noise, or overheating in rotational assembly.\n"
         "Action Required: Immediately isolate machine power. Remove bearing housing and replace worn ball bearings with part #BRG-9920. "
         "Apply high-temperature synthetic grease prior to restarting operations. Priority: High.")
    ],
    
    "structural_inspection_sop.pdf": [
        ("Page 1: Visual Structural Audits", 
         "Guidelines for structural integrity audits across industrial frames, concrete supports, and metallic enclosures. "
         "Inspect critical load-bearing joints every 30 days."),
        
        ("Page 2: Structural Crack Mitigation", 
         "Defect Type: Structural Crack.\n"
         "Symptom: Linear fracture propagation on structural casing or welded support seams.\n"
         "Action Required: Halt structural loads immediately. Perform ultrasonic non-destructive testing (NDT) to assess depth. "
         "V-groove the crack and execute a full-penetration weld repair using E7018 electrodes. Reinforce joint with structural backing plate. Priority: Critical.")
    ],
    
    "corrosion_control_guide.pdf": [
        ("Page 1: Environmental Corrosion Standards", 
         "Overview of atmospheric oxidation and corrosion prevention in fluid transmission lines and structural steel."),
        
        ("Page 2: Rust & Oxidation Treatment", 
         "Defect Type: Surface Corrosion and Rust.\n"
         "Symptom: Brownish oxide flakes, surface pitting, or paint degradation on metallic piping.\n"
         "Action Required: Scrape loose rust off using wire brush or sandblasting. Apply chemical rust converter followed by two coats of zinc-rich primer paint. "
         "Inspect wall thickness with ultrasonic gauge if pitting exceeds 1mm depth. Priority: Medium.")
    ]
}

def create_pdfs():
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=16, leading=20)
    body_style = ParagraphStyle('BodyStyle', parent=styles['Normal'], fontSize=11, leading=15)

    print("Generating Oct 5-6 sample maintenance manuals (PDFs)...")

    for filename, pages in MANUALS.items():
        pdf_path = os.path.join(RAW_DATA_DIR, filename)
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        story = []

        for i, (heading, text) in enumerate(pages):
            story.append(Paragraph(heading, title_style))
            story.append(Spacer(1, 12))
            for paragraph in text.split('\n'):
                story.append(Paragraph(paragraph, body_style))
                story.append(Spacer(1, 8))
            
            if i < len(pages) - 1:
                story.append(PageBreak())  # Force explicit page break for exact page testing

        doc.build(story)
        print(f" Created {filename} ({len(pages)} pages) -> {pdf_path}")

    print(f"\nManual generation complete! PDFs stored in: {RAW_DATA_DIR}")

if __name__ == "__main__":
    create_pdfs()