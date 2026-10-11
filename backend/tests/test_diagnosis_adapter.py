import pytest

from app.diagnosis_adapter import DiagnosisFormatError, normalize_agent_output

GOOD = {
    "defect_type": "Conveyor belt fraying", "severity": "High", "explanation": "Edge is frayed.",
    "recommendation": "Replace the belt tensioner.", "manual": "Conveyor Manual",
    "page": "p. 14", "confidence": 87,
}


def test_messy_but_valid_output_is_cleaned():
    d = normalize_agent_output(GOOD, location="Line 3", image_ref="belt.png")
    assert d.defect == "Conveyor belt fraying"
    assert d.severity == "high"            # "High" -> "high"
    assert d.source_page == 14             # "p. 14" -> 14
    assert d.confidence == 0.87            # 87 -> 0.87
    assert d.raw["location"] == "Line 3" and d.raw["image_ref"] == "belt.png"


@pytest.mark.parametrize("given,expected", [("Severe", "high"), ("minor", "low"), ("URGENT", "critical"), ("moderate", "medium")])
def test_severity_synonyms(given, expected):
    assert normalize_agent_output({**GOOD, "severity": given}).severity == expected


@pytest.mark.parametrize("given,expected", [(0.5, 0.5), (87, 0.87), ("87%", 0.87), (None, None), (1, 1.0)])
def test_confidence_forms(given, expected):
    assert normalize_agent_output({**GOOD, "confidence": given}).confidence == expected


def test_unknown_severity_is_an_error_not_a_guess():
    with pytest.raises(DiagnosisFormatError) as e:
        normalize_agent_output({**GOOD, "severity": "scary"})
    assert any("severity" in p for p in e.value.problems)


def test_missing_citation_is_an_error():
    bad = {k: v for k, v in GOOD.items() if k not in ("manual", "page")}
    with pytest.raises(DiagnosisFormatError) as e:
        normalize_agent_output(bad)
    assert len(e.value.problems) == 2  # manual AND page reported together


def test_confidence_out_of_range_is_an_error():
    with pytest.raises(DiagnosisFormatError):
        normalize_agent_output({**GOOD, "confidence": 250})
    with pytest.raises(DiagnosisFormatError):
        normalize_agent_output({**GOOD, "confidence": "high"})


def test_result_goes_through_the_service(client):
    """normalize -> attach_diagnosis works end to end and the case becomes approvable."""
    from tests.helpers import approve, upload_case
    cid = upload_case(client)
    diag = normalize_agent_output(GOOD, location="Line 3", image_ref="belt.png")
    client.app.state.service.attach_diagnosis(cid, diag)
    assert approve(client, cid).json()["status"] == "task_created"
