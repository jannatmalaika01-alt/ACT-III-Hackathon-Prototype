"""The citation check inside the Evolus flow: every quote in the agent's reply is checked, the case stores the checked list, and a
quote that is not in our manuals sets the human review flag. HTTP is mocked; the manuals are the real corrosion-guide pages."""
import pytest

from app import citations
from tests.test_evolus_rest import CASE, Resp, client, execution, http  # noqa: F401  (fixtures)

P3 = ("Take the line out of pressurised service. Cut out and replace the affected pipe section when wall loss exceeds 20 percent; "
      "between 10 and 20 percent schedule repair within 7 days and re-measure monthly. Priority: High.")


def reply(citation_lines, needs_review="false"):
    cites = "\n".join(f"- {c}" for c in citation_lines)
    return (f"**defect:** Corrosion\n**severity:** High\n**probable_cause:** Moisture.\n**safety_precautions:**\n- Isolate the line\n"
            f"**recommended_steps:**\n1. Scrape the rust\n**citation:**\n{cites}\n**confidence:** high\n"
            f"**needs_human_review:** {needs_review}\n**summary:** Flange corrosion.")


def cite(page, quote, doc="corrosion_control_guide.txt"):
    return f'document: {doc} — section: Page {page}: Title — quote: "{quote}"'


GOOD = cite(3, P3)
SPELLED = cite(3, P3.replace("pressurised", "pressurized"))
INVENTED = cite(2, "Always replace the whole pipe network immediately and tell nobody.")


def run(http, lines, needs_review="false"):
    http["polls"] = [Resp(200, execution("Completed", reply(lines, needs_review)))]
    return client().create_task(CASE).result


def test_every_quote_gets_a_status_and_the_checked_list_is_stored_in_the_result(http):
    r = run(http, [GOOD, SPELLED, INVENTED])
    assert [c["status"] for c in r["citations"]] == ["verified", "corrected", "unverified"]
    assert r["citation_check"] == {"verified": 1, "corrected": 1, "unverified": 1}
    good, spelled, invented = r["citations"]
    assert (good["document"], good["page"]) == ("corrosion_control_guide.pdf", 3) and good["quote"] == P3
    assert "pressurised" in spelled["quote"] and "pressurized" in spelled["agent_quote"]          # replaced by the manual's words
    assert invented["quote"].startswith("Always replace the whole pipe network") and invented["note"]    # the agent's text is kept
    assert r["parsed"]["citation"][2] == INVENTED                                                 # the agent's own list is untouched


def test_an_unverified_quote_sets_the_human_review_flag_and_keeps_what_the_agent_said(http):
    r = run(http, [GOOD, INVENTED], needs_review="false")
    assert r["parsed"]["needs_human_review"] is True and r["agent_needs_human_review"] is False
    assert any("1 citation(s) could not be verified against our manuals" in n for n in r["notes"])


def test_corrected_and_verified_quotes_do_not_change_the_agents_own_flag(http):
    ok = run(http, [GOOD, SPELLED], needs_review="false")
    assert ok["parsed"]["needs_human_review"] is False and ok["agent_needs_human_review"] is False
    assert any("1 quote(s) were corrected" in n for n in ok["notes"])
    asked = run(http, [GOOD], needs_review="true")
    assert asked["parsed"]["needs_human_review"] is True and asked["agent_needs_human_review"] is True    # the agent's own request stays


def test_no_citation_at_all_is_noted(http):
    r = run(http, [])
    assert r["citations"] == [] and r["citation_check"] == {"verified": 0, "corrected": 0, "unverified": 0}
    assert "the agent gave no citation" in r["notes"]


def test_a_failing_check_never_breaks_the_task_and_leaves_the_citations_unverified(http, monkeypatch):
    def boom():
        raise RuntimeError("index unavailable")

    monkeypatch.setattr(citations, "load_passages", boom)
    monkeypatch.setattr(citations, "_PASSAGES", None)
    r = run(http, [GOOD])
    assert r["citations"][0]["status"] == "unverified" and r["parsed"]["needs_human_review"] is True and r["execution_id"]


# ---------------- through the case ----------------

@pytest.fixture
def app_client(http):
    from fastapi.testclient import TestClient
    from app.config import Settings
    from app.db import InMemoryRepo
    from app.main import create_app
    settings = Settings(supabase_url="", supabase_key="", storage_bucket="b", webhook_secret="s", evolus_mode="rest", evolus_mock_fail_first_n=0,
                        max_upload_mb=1, use_memory_db=True, evolus_background=False)
    tc = TestClient(create_app(settings=settings, repo=InMemoryRepo(), evolus=client()))
    case = tc.post("/cases", files={"file": ("a.png", b"x", "image/png")}, data={"uploaded_by": "t"}).json()
    tc.post(f"/cases/{case['id']}/diagnosis", json={"defect": "corrosion", "severity": "high", "explanation": "e", "recommended_action": "a",
                                                    "source_manual": "m.pdf", "source_page": 1, "confidence": 0.9,
                                                    "raw": {"defect_report": {"defect_type": "corrosion", "location": "flange"}}})
    return tc, case["id"]


def test_the_case_stores_the_verified_citation_list_and_the_event_says_so(app_client, http):
    tc, cid = app_client
    http["polls"] = [Resp(200, execution("Completed", reply([GOOD, SPELLED, INVENTED])))]
    body = tc.post(f"/cases/{cid}/approve", json={"decided_by": "boss"}).json()
    ev = body["diagnosis_raw"]["evolus"]
    assert body["status"] == "task_created" and [c["status"] for c in ev["citations"]] == ["verified", "corrected", "unverified"]
    assert ev["parsed"]["needs_human_review"] is True and ev["agent_needs_human_review"] is False
    notes = [e["note"] for e in tc.get(f"/cases/{cid}").json()["events"]]
    assert any("1 citation(s) NOT verified against the manuals" in (n or "") and "needs_human_review=True" in n for n in notes)


def test_a_clean_answer_adds_nothing_to_the_event(app_client, http):
    tc, cid = app_client
    http["polls"] = [Resp(200, execution("Completed", reply([GOOD])))]
    tc.post(f"/cases/{cid}/approve", json={"decided_by": "boss"})
    notes = [e["note"] for e in tc.get(f"/cases/{cid}").json()["events"]]
    assert not any("NOT verified" in (n or "") for n in notes) and any("needs_human_review=False" in (n or "") for n in notes)

