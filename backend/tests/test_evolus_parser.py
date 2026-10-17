"""parse_agent_reply: the Evolus agent answers in free text; every markdown style must still give the same fields."""
import json

import pytest

from app.evolus import FIELDS, parse_agent_reply

EXPECTED_STEPS = ["Inspect the flange face for pitting", "Replace the gasket and re-torque the bolts"]
EXPECTED_SAFETY = ["Isolate and depressurise the line", "Wear gloves and eye protection"]

BOLD_COLON_INSIDE = """**defect:** Corrosion on pipe flange
**severity:** High
**probable_cause:** Moisture ingress and a failed gasket.
**safety_precautions:**
- Isolate and depressurise the line
- Wear gloves and eye protection
**recommended_steps:**
1. Inspect the flange face for pitting
2. Replace the gasket and re-torque the bolts
**citation:** corrosion_control_guide.pdf, page 2
**confidence:** 0.85
**needs_human_review:** false
**summary:** Moderate flange corrosion."""

BOLD_COLON_OUTSIDE = """**Defect**: Corrosion on pipe flange
**Severity**: High
**Probable cause**: Moisture ingress and a failed gasket.
**Safety precautions**:
* Isolate and depressurise the line
* Wear gloves and eye protection
**Recommended steps**:
- Inspect the flange face for pitting
- Replace the gasket and re-torque the bolts
**Citation**: corrosion_control_guide.pdf, page 2
**Confidence**: 85%
**Needs human review**: No
**Summary**: Moderate flange corrosion."""

PLAIN = """defect: Corrosion on pipe flange
severity: high
probable_cause: Moisture ingress and a failed gasket.
safety_precautions:
 - Isolate and depressurise the line
 - Wear gloves and eye protection
recommended_steps:
 - Inspect the flange face for pitting
 - Replace the gasket and re-torque the bolts
citation: corrosion_control_guide.pdf, page 2
confidence: 0.85
needs_human_review: no
summary: Moderate flange corrosion."""

HEADINGS = """## Defect
Corrosion on pipe flange
## Severity
High
### Probable cause
Moisture ingress and a failed gasket.
### Safety precautions
- Isolate and depressurise the line
- Wear gloves and eye protection
### Recommended steps
1. Inspect the flange face for pitting
2. Replace the gasket and re-torque the bolts
## Citation
corrosion_control_guide.pdf, page 2
## Confidence
0.85
## Needs human review
False
## Summary
Moderate flange corrosion."""

BULLETED_FIELDS = """- **defect:** Corrosion on pipe flange
- **severity:** High
- **probable_cause:** Moisture ingress and a failed gasket.
- **safety_precautions:** Isolate and depressurise the line
  - Wear gloves and eye protection
- **recommended_steps:** Inspect the flange face for pitting
  - Replace the gasket and re-torque the bolts
- **citation:** corrosion_control_guide.pdf, page 2
- **confidence:** 0.85
- **needs_human_review:** false
- **summary:** Moderate flange corrosion."""

AS_JSON = json.dumps({"defect": "Corrosion on pipe flange", "severity": "High", "probable_cause": "Moisture ingress and a failed gasket.",
                      "safety_precautions": EXPECTED_SAFETY, "recommended_steps": EXPECTED_STEPS,
                      "citation": "corrosion_control_guide.pdf, page 2", "confidence": 0.85, "needs_human_review": False,
                      "summary": "Moderate flange corrosion."})


@pytest.mark.parametrize("text", [BOLD_COLON_INSIDE, BOLD_COLON_OUTSIDE, PLAIN, HEADINGS, BULLETED_FIELDS, AS_JSON,
                                  "```json\n" + AS_JSON + "\n```", "Here is my assessment:\n\n" + BOLD_COLON_INSIDE + "\n\nStay safe!"])
def test_every_style_gives_the_same_fields(text):
    r = parse_agent_reply(text)
    p = r["parsed"]
    assert p["defect"] == "Corrosion on pipe flange" and p["severity"] == "high"
    assert p["probable_cause"] == "Moisture ingress and a failed gasket."
    assert p["safety_precautions"] == EXPECTED_SAFETY and p["recommended_steps"] == EXPECTED_STEPS
    assert p["citation"] == ["corrosion_control_guide.pdf, page 2"]
    assert p["confidence"] == 0.85 and p["needs_human_review"] is False
    assert p["summary"] == "Moderate flange corrosion."
    assert r["raw_text"] == text and r["notes"] == []                    # the raw text is always kept
    assert set(p) == set(FIELDS) | {"confidence_text"}


@pytest.mark.parametrize("value,expected", [("true", True), ("True", True), ("yes", True), ("Yes - the cause is unclear", True),
                                            ("required", True), ("1", True), ("false", False), ("False", False), ("no", False),
                                            ("No", False), ("not needed", False), ("**false**", False), ("false.", False)])
def test_needs_human_review_becomes_a_boolean(value, expected):
    assert parse_agent_reply(f"**needs_human_review:** {value}")["parsed"]["needs_human_review"] is expected


def test_missing_or_unclear_review_flag_defaults_to_true_with_a_note():
    for text in ("**defect:** x", "**needs_human_review:** maybe", ""):
        r = parse_agent_reply(text)
        assert r["parsed"]["needs_human_review"] is True and any("needs_human_review" in n for n in r["notes"])


@pytest.mark.parametrize("value,expected", [("0.85", 0.85), ("85%", 0.85), ("85", 0.85), ("1", 1.0), ("0", 0.0),
                                            ("0.9 (high)", 0.9), ("high", None), ("150", None)])
def test_confidence(value, expected):
    assert parse_agent_reply(f"**confidence:** {value}")["parsed"]["confidence"] == expected


@pytest.mark.parametrize("word", ["high", "High", "medium", "low", "**high**"])
def test_a_word_for_confidence_is_kept_as_text_without_a_complaint(word):
    r = parse_agent_reply(f"**confidence:** {word}")
    assert r["parsed"]["confidence"] is None and r["parsed"]["confidence_text"].strip("*").lower() == word.strip("*").lower()
    assert not any("confidence" in n and "neither" in n for n in r["notes"])


def test_an_unreadable_confidence_is_noted():
    r = parse_agent_reply("**confidence:** somewhat sure")
    assert r["parsed"]["confidence"] is None and any("neither a number nor high/medium/low" in n for n in r["notes"])


@pytest.mark.parametrize("value,expected", [("High", "high"), ("**CRITICAL**", "critical"), ("medium - monitor", "medium"),
                                            ("Low", "low"), ("urgent", "urgent")])
def test_severity_is_normalised_when_it_is_one_of_the_four(value, expected):
    assert parse_agent_reply(f"**severity:** {value}")["parsed"]["severity"] == expected


def test_list_fields_without_bullets_become_a_one_item_list():
    p = parse_agent_reply("**safety_precautions:** Wear gloves.\n**recommended_steps:** Replace the gasket.")["parsed"]
    assert p["safety_precautions"] == ["Wear gloves."] and p["recommended_steps"] == ["Replace the gasket."]


def test_a_step_that_looks_like_a_field_name_stays_a_step():
    p = parse_agent_reply("**recommended_steps:**\n- Isolate: close the upstream valve\n- Inspect: check the gasket")["parsed"]
    assert p["recommended_steps"] == ["Isolate: close the upstream valve", "Inspect: check the gasket"]


def test_multi_line_values_are_joined():
    p = parse_agent_reply("**summary:** Line one\nand line two\n**defect:** x")["parsed"]
    assert p["summary"] == "Line one and line two" and p["defect"] == "x"


def test_garbage_and_non_text_never_raise():
    for junk in ("", "   ", "no fields here at all", None, 42, "{not json", "**:**", "###"):
        r = parse_agent_reply(junk)
        assert set(r["parsed"]) == set(FIELDS) | {"confidence_text"} and r["parsed"]["needs_human_review"] is True
    assert parse_agent_reply("no fields here")["raw_text"] == "no fields here"


def test_missing_fields_are_listed_in_the_notes():
    r = parse_agent_reply("**defect:** x\n**needs_human_review:** false")
    assert "severity not found in the reply" in r["notes"] and r["parsed"]["severity"] is None


# The agent's real reply from the live end-to-end run (Evolus execution dbdee6ac-33ca-f111-bf5b-6045bd9ce552).
REAL_REPLY = """**defect:** Surface corrosion and rust (brownish oxide flakes / surface pitting / paint degradation on metallic piping) — location not specified in upload

**severity:** medium

**probable_cause:** Surface oxidation of metallic piping, consistent with the symptom described in the corrosion guide.

**safety_precautions:**
- General practice, not from the library: apply lockout-tagout and isolate the line before any physical work.
- General practice, not from the library: wear PPE (gloves, eye protection, respiratory protection for sandblasting).

**recommended_steps:**
1. Scrape loose rust off using wire brush or sandblasting.
2. Apply chemical rust converter followed by two coats of zinc-rich primer paint.
3. Inspect wall thickness with ultrasonic gauge if pitting exceeds 1mm depth.
4. If wall loss above 20 percent of nominal thickness is found, take the line out of pressurised service and cut out/replace the affected pipe section; between 10 and 20 percent, schedule repair within 7 days and re-measure monthly.
5. Re-coat repaired steel within 4 hours of surface preparation; record coating thickness readings after each paint job.

**citation:**
- document: corrosion_control_guide.txt — section: Page 2: Rust & Oxidation Treatment — quote: "Action Required: Scrape loose rust off using wire brush or sandblasting. Priority: Medium."
- document: corrosion_control_guide.txt — section: Page 3: Pitting and Wall Thickness Loss — quote: "Take the line out of pressurised service. Priority: High."
- document: corrosion_control_guide.txt — section: Page 4: Protective Coatings and Cathodic Protection — quote: "Re-coat repaired steel within 4 hours of surface preparation."

**confidence:** high

**needs_human_review:** false

**summary:** Both vision models verify surface corrosion; follow the corrosion guide's rust-removal, zinc-rich primer and ultrasonic wall-thickness check steps (CNN's structural_crack is low-trust advisory only)."""


def test_the_real_agent_reply_from_the_live_run():
    r = parse_agent_reply(REAL_REPLY)
    p = r["parsed"]
    assert p["defect"].startswith("Surface corrosion and rust") and p["severity"] == "medium"
    assert p["probable_cause"].startswith("Surface oxidation of metallic piping")
    assert len(p["safety_precautions"]) == 2 and p["safety_precautions"][0].startswith("General practice, not from the library")
    assert len(p["recommended_steps"]) == 5 and p["recommended_steps"][0] == "Scrape loose rust off using wire brush or sandblasting."
    assert len(p["citation"]) == 3 and p["citation"][0].startswith("document: corrosion_control_guide.txt")
    assert p["confidence"] is None and p["confidence_text"] == "high"
    assert p["needs_human_review"] is False and p["summary"].startswith("Both vision models verify surface corrosion")
    assert r["notes"] == [] and r["raw_text"] == REAL_REPLY
