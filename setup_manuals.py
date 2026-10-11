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
         "Apply high-temperature synthetic grease prior to restarting operations. Priority: High."),

        ("Page 3: Bearing Overheating and Lubrication Failure",
         "Defect Type: Bearing Overheating.\n"
         "Symptom: Housing temperature above 95 C, discoloured (blue or brown) races, burnt grease smell, grease leaking past the seals.\n"
         "Action Required: Stop the machine and let it cool for 30 minutes. Purge old grease, inspect the seals and re-lubricate with grease type HT-2. "
         "If races are discoloured, replace the bearing set. Do not exceed 30 percent grease fill. Priority: High."),

        ("Page 4: Bearing Installation, Fit and Torque",
         "Procedure: Installing a replacement bearing.\n"
         "Heat the inner ring with an induction heater to between 80 and 100 C; never use an open flame. Slide the bearing onto the shaft without hammering. "
         "Tighten the lock nut to 85 Nm and bend the lock washer tab. Check shaft runout is below 0.02 mm with a dial gauge. Priority: Medium."),

        ("Page 5: Spare Parts and Maintenance Records",
         "Keep two spare bearing sets (part #BRG-9920) in the stores. Record every bearing change in the maintenance log with date, "
         "running hours, operator and the failure mode found. Review the log monthly to find repeat failures."),
    ],

    "structural_inspection_sop.pdf": [
        ("Page 1: Visual Structural Audits",
         "Guidelines for structural integrity audits across industrial frames, concrete supports, and metallic enclosures. "
         "Inspect critical load-bearing joints every 30 days."),

        ("Page 2: Structural Crack Mitigation",
         "Defect Type: Structural Crack.\n"
         "Symptom: Linear fracture propagation on structural casing or welded support seams.\n"
         "Action Required: Halt structural loads immediately. Perform ultrasonic non-destructive testing (NDT) to assess depth. "
         "V-groove the crack and execute a full-penetration weld repair using E7018 electrodes. Reinforce joint with structural backing plate. Priority: Critical."),

        ("Page 3: Weld Repair Verification",
         "Procedure: Checking a repaired weld.\n"
         "After a weld repair let the joint cool to ambient temperature, then run dye penetrant or magnetic particle testing over the full seam. "
         "Record the crack length before and after repair. A repair that shows any remaining linear indication must be ground out and repeated. Priority: High."),

        ("Page 4: Re-commissioning and Sign-off",
         "Procedure: Returning a structure to service.\n"
         "A certified inspector must sign the re-commissioning form after NDT passes. Apply load in 25 percent steps and watch the repaired joint "
         "for new cracking. Update the inspection schedule to every 7 days for the next 90 days."),
    ],

    "corrosion_control_guide.pdf": [
        ("Page 1: Environmental Corrosion Standards",
         "Overview of atmospheric oxidation and corrosion prevention in fluid transmission lines and structural steel."),

        ("Page 2: Rust & Oxidation Treatment",
         "Defect Type: Surface Corrosion and Rust.\n"
         "Symptom: Brownish oxide flakes, surface pitting, or paint degradation on metallic piping.\n"
         "Action Required: Scrape loose rust off using wire brush or sandblasting. Apply chemical rust converter followed by two coats of zinc-rich primer paint. "
         "Inspect wall thickness with ultrasonic gauge if pitting exceeds 1mm depth. Priority: Medium."),

        ("Page 3: Pitting and Wall Thickness Loss",
         "Defect Type: Corrosion with wall thinning.\n"
         "Symptom: Deep pits, or ultrasonic readings showing wall loss above 20 percent of nominal thickness.\n"
         "Action Required: Take the line out of pressurised service. Cut out and replace the affected pipe section when wall loss exceeds 20 percent; "
         "between 10 and 20 percent schedule repair within 7 days and re-measure monthly. Priority: High."),

        ("Page 4: Protective Coatings and Cathodic Protection",
         "Procedure: Preventing corrosion returning.\n"
         "Re-coat repaired steel within 4 hours of surface preparation. Check sacrificial anodes every 6 months and replace any that have lost more than half their mass. "
         "Record coating thickness readings after each paint job."),
    ],

    "fastener_alignment_sop.pdf": [
        ("Page 1: Fasteners and Alignment Overview",
         "Loose fasteners and misaligned shafts or conveyor belts cause vibration and premature wear. "
         "Check torque marks and belt tracking during every weekly round."),

        ("Page 2: Loose Bolt Re-tightening",
         "Defect Type: Loose Bolt.\n"
         "Symptom: Broken torque stripe, rattling, visible gap under the bolt head or fretting dust around a flange.\n"
         "Action Required: Re-tighten to the specified torque in a cross pattern; for M16 bolts the value is 210 Nm. Replace any bolt that is stretched or has damaged threads "
         "and issue an immediate tightening work order. Priority: Medium."),

        ("Page 3: Shaft and Conveyor Misalignment",
         "Defect Type: Misalignment.\n"
         "Symptom: Belt drifting to one side, uneven wear, high axial vibration or coupling heat.\n"
         "Action Required: Isolate the drive, align shafts with a laser alignment tool to within 0.05 mm offset and re-track the belt. "
         "If misalignment recurs more than twice in 24 hours call a maintenance technician. Priority: Medium."),
    ],

    "condition_monitoring_guide.pdf": [
        ("Page 1: Vibration Limits",
         "Vibration severity guide for rotating machines. Below 4.5 mm/s RMS the machine is in good condition. Between 4.5 and 7.1 mm/s raise a warning and plan an inspection. "
         "Above 7.1 mm/s treat it as an alarm: suspect bearing wear or imbalance and inspect the bearings the same shift."),

        ("Page 2: Temperature Monitoring and Thermography",
         "Bearing housing temperature above 80 C is a warning, above 95 C is an alarm. Use an infrared camera monthly and compare against the baseline image. "
         "A temperature rise of more than 15 C in a week with rising vibration points to lubrication failure or bearing wear."),

        ("Page 3: Ultrasonic Thickness Monitoring",
         "Measure wall thickness at fixed test points every 6 months. Calculate wall loss as a percentage of nominal thickness. "
         "Ten percent loss is a warning and twenty percent is an alarm that requires corrosion repair or section replacement."),
    ],
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