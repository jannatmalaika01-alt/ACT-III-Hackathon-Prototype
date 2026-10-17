"""The citation verifier: every quote the Evolus agent gives is looked up in our manuals."""
import pytest

from app import citations as c
from tests.conftest import BEARING_P2, CORROSION_P2, CORROSION_P3, CORROSION_P4

PASSAGES = [c.Passage("corrosion_control_guide.pdf", 2, CORROSION_P2), c.Passage("corrosion_control_guide.pdf", 3, CORROSION_P3),
            c.Passage("corrosion_control_guide.pdf", 4, CORROSION_P4), c.Passage("bearing_maintenance_manual.pdf", 2, BEARING_P2)]


def line(doc, page, quote, title="Pitting"):
    """A citation line the way the Evolus agent writes it."""
    return f'document: {doc} — section: Page {page}: {title} — quote: "{quote}"'


EXACT_P3 = ("Take the line out of pressurised service. Cut out and replace the affected pipe section when wall loss exceeds 20 percent; "
            "between 10 and 20 percent schedule repair within 7 days and re-measure monthly. Priority: High.")


# ---------------- small helpers ----------------

def test_normalize_ignores_case_punctuation_and_whitespace():
    assert c.normalize("  Re-measure   MONTHLY.\n Priority: High! ") == "re measure monthly priority high"
    assert c.normalize("“Quoted” — dash; (brackets)") == "quoted dash brackets" and c.normalize("") == c.normalize(None) == ""


def test_stem_maps_txt_and_pdf_to_the_same_name():
    assert c.stem("corrosion_control_guide.txt") == c.stem("Corrosion_Control_Guide.PDF") == "corrosion_control_guide"


def test_parse_the_agents_line_in_its_forms():
    p = c.parse_item(line("corrosion_control_guide.txt", 3, "Take the line out.", "Pitting and Wall Thickness Loss"))
    assert (p["document"], p["page"], p["quote"]) == ("corrosion_control_guide.txt", 3, "Take the line out.")
    assert c.parse_item('document: a.txt - quote: "Replace it"')["quote"] == "Replace it"
    plain = c.parse_item("corrosion_control_guide.pdf, page 2")
    assert (plain["document"], plain["quote"]) == ("corrosion_control_guide.pdf", None)
    d = c.parse_item({"document": "a.txt", "page": "4", "quote": "x y z"})
    assert (d["document"], d["page"], d["quote"]) == ("a.txt", 4, "x y z")


# ---------------- the three outcomes ----------------

def test_an_exact_quote_is_verified_and_the_pdf_name_replaces_the_txt_name():
    r = c.verify_citation(line("corrosion_control_guide.txt", 3, EXACT_P3), PASSAGES)
    assert (r["status"], r["document"], r["page"], r["similarity"]) == ("verified", "corrosion_control_guide.pdf", 3, 1.0)
    assert r["quote"] == EXACT_P3 and "agent_quote" not in r and "claimed" not in r and "note" not in r


def test_whitespace_case_and_punctuation_differences_still_verify():
    sloppy = "take   the line out of pressurised service cut out and replace the affected\npipe section WHEN wall loss exceeds 20 percent"
    r = c.verify_citation(line("corrosion_control_guide.txt", 3, sloppy), PASSAGES)
    assert r["status"] == "verified" and r["quote"] == sloppy                  # the agent's own text is kept when it matches


def test_pressurized_for_pressurised_is_corrected_to_the_manuals_exact_text():
    agent = ("Take the line out of pressurized service. Cut out and replace the affected pipe section when wall loss exceeds 20 percent; "
             "between 10 and 20 percent schedule repair within 7 days and re-measure monthly. Priority: High.")
    r = c.verify_citation(line("corrosion_control_guide.txt", 3, agent), PASSAGES)
    assert r["status"] == "corrected" and 0.9 <= r["similarity"] < 1.0
    assert r["quote"] == EXACT_P3 and "pressurised" in r["quote"]              # replaced by the manual's own words, punctuation included
    assert r["agent_quote"] == agent and "pressurized" in r["agent_quote"]     # what the agent wrote is kept
    assert (r["document"], r["page"]) == ("corrosion_control_guide.pdf", 3) and "claimed" not in r


def test_a_missing_word_is_corrected_too():
    agent = "Take the line out of pressurised service. Cut out and replace the pipe section when wall loss exceeds 20 percent"
    r = c.verify_citation(line("corrosion_control_guide.txt", 3, agent), PASSAGES)
    assert r["status"] == "corrected" and "the affected pipe section" in r["quote"] and r["agent_quote"] == agent


def test_a_spelling_change_in_a_short_quote_is_corrected():
    r = c.verify_citation(line("corrosion_control_guide.txt", 4, "Re-coat repaired steel within 4 hours of surface preperation."), PASSAGES)
    assert r["status"] == "corrected" and r["quote"] == "Re-coat repaired steel within 4 hours of surface preparation."


def test_an_invented_quote_is_unverified_and_keeps_the_agents_text():
    invented = "Replace the entire pipe network with titanium panels and notify the regulator within one hour."
    r = c.verify_citation(line("corrosion_control_guide.txt", 3, invented), PASSAGES)
    assert r["status"] == "unverified" and r["quote"] == invented and r["similarity"] < 0.9
    assert r["note"] == "this quote is not in our manuals" and "agent_quote" not in r
    assert (r["document"], r["page"]) == ("corrosion_control_guide.pdf", 3)         # still names the real file the agent pointed at


def test_a_citation_without_a_quote_cannot_be_checked():
    r = c.verify_citation("corrosion_control_guide.pdf, page 2", PASSAGES)
    assert r["status"] == "unverified" and r["note"] == "no quote to check" and r["document"] == "corrosion_control_guide.pdf"


# ---------------- where it looks ----------------

def test_a_wrong_page_is_found_on_the_right_page_and_says_so():
    r = c.verify_citation(line("corrosion_control_guide.txt", 2, EXACT_P3), PASSAGES)         # the quote is on page 3, not 2
    assert (r["status"], r["page"]) == ("verified", 3)
    assert r["claimed"] == {"document": "corrosion_control_guide.txt", "page": 2} and "page 3" in r["note"]


def test_a_wrong_document_is_found_in_the_right_manual_and_says_so():
    r = c.verify_citation(line("bearing_maintenance_manual.txt", 2, EXACT_P3), PASSAGES)
    assert (r["status"], r["document"], r["page"]) == ("verified", "corrosion_control_guide.pdf", 3)
    assert r["claimed"]["document"] == "bearing_maintenance_manual.txt" and "corrosion_control_guide.pdf" in r["note"]


def test_an_unknown_document_is_searched_everywhere():
    r = c.verify_citation(line("some_other_manual.txt", 9, EXACT_P3), PASSAGES)
    assert (r["status"], r["document"], r["page"]) == ("verified", "corrosion_control_guide.pdf", 3)
    bad = c.verify_citation(line("some_other_manual.txt", 9, "Nothing like any manual text at all."), PASSAGES)
    assert (bad["status"], bad["document"]) == ("unverified", "some_other_manual.txt")      # the unknown name is kept as written


def test_the_right_page_wins_over_a_similar_page():
    twin = c.Passage("corrosion_control_guide.pdf", 7, CORROSION_P3.replace("pressurised", "pressurized"))
    r = c.verify_citation(line("corrosion_control_guide.txt", 3, EXACT_P3), PASSAGES + [twin])
    assert (r["status"], r["page"]) == ("verified", 3)


# ---------------- the threshold ----------------

def test_the_similarity_threshold_is_0_9_and_can_be_changed():
    agent = "Take the line out of pressurized service. Cut out and replace the affected pipe section when wall loss exceeds 20 percent"
    item = line("corrosion_control_guide.txt", 3, agent)
    assert c.THRESHOLD == 0.90 and c.verify_citation(item, PASSAGES)["status"] == "corrected"
    assert c.verify_citation(item, PASSAGES, threshold=0.999)["status"] == "unverified"       # stricter: the same quote now fails
    far = c.verify_citation(line("corrosion_control_guide.txt", 3, "Take the line out of pressurized service. Cut out and swap the affected duct."), PASSAGES)
    assert far["status"] == "unverified"                                                       # about 0.8: below the line


# ---------------- the manuals themselves ----------------

def test_pages_are_rebuilt_from_overlapping_chunks_so_a_quote_across_a_boundary_matches():
    chunks = [{"source": "m.pdf", "page": 1, "text": "Alpha one is here. Bravo two follows. Charlie three ends."},
              {"source": "m.pdf", "page": 1, "text": "Charlie three ends. Delta four starts. Echo five finishes."},   # starts with the overlap sentence
              {"source": "m.pdf", "page": 2, "text": "Another page."}]
    ps = c.passages_from_chunks(chunks)
    assert [(p.source, p.page) for p in ps] == [("m.pdf", 1), ("m.pdf", 2)]
    assert ps[0].text.count("Charlie three ends") == 1
    r = c.verify_citation(line("m.txt", 1, "Bravo two follows. Charlie three ends. Delta four starts."), ps)
    assert r["status"] == "verified" and r["document"] == "m.pdf"


def test_summarize_counts_every_status():
    items = [line("corrosion_control_guide.txt", 3, EXACT_P3), line("corrosion_control_guide.txt", 3, EXACT_P3.replace("pressurised", "pressurized")),
             line("corrosion_control_guide.txt", 3, "Invented text that is nowhere.")]
    assert c.summarize(c.verify_citations(items, PASSAGES)) == {"verified": 1, "corrected": 1, "unverified": 1}


def test_if_the_manuals_cannot_be_loaded_every_citation_is_left_unverified(monkeypatch):
    def boom():
        raise RuntimeError("index unavailable")

    monkeypatch.setattr(c, "load_passages", boom)
    out = c.verify_citations([line("corrosion_control_guide.txt", 3, EXACT_P3)])
    assert out[0]["status"] == "unverified" and "could not be loaded" in out[0]["note"] and out[0]["quote"] == EXACT_P3


# ---------------- against the REAL manuals of the retrieval index ----------------

LIVE = [  # the three citations the live Evolus agent wrote for img_18 (execution dbdee6ac-33ca-f111-bf5b-6045bd9ce552)
    line("corrosion_control_guide.txt", 2, "Action Required: Scrape loose rust off using wire brush or sandblasting. Apply chemical rust converter "
         "followed by two coats of zinc-rich primer paint. Inspect wall thickness with ultrasonic gauge if pitting exceeds 1mm depth. Priority: Medium.",
         "Rust & Oxidation Treatment"),
    line("corrosion_control_guide.txt", 3, "Take the line out of pressurised service. Cut out and replace the affected pipe section when wall loss "
         "exceeds 20 percent; between 10 and 20 percent schedule repair within 7 days and re-measure monthly. Priority: High.", "Pitting and Wall Thickness Loss"),
    line("corrosion_control_guide.txt", 4, "Re-coat repaired steel within 4 hours of surface preparation. Check sacrificial anodes every 6 months and "
         "replace any that have lost more than half their mass. Record coating thickness readings after each paint job.", "Protective Coatings"),
]


@pytest.fixture
def real_index(monkeypatch):
    monkeypatch.setattr(c, "_PASSAGES", None)       # use the real index built from data/raw
    return c.load_passages()


def test_the_real_index_holds_the_five_manuals(real_index):
    assert {p.source for p in real_index} == {"bearing_maintenance_manual.pdf", "condition_monitoring_guide.pdf", "corrosion_control_guide.pdf",
                                              "fastener_alignment_sop.pdf", "structural_inspection_sop.pdf"}


def test_the_live_agents_quotes_verify_against_the_real_manuals(real_index):
    out = c.verify_citations(LIVE, real_index)
    assert [r["status"] for r in out] == ["verified", "verified", "verified"]
    assert [(r["document"], r["page"]) for r in out] == [("corrosion_control_guide.pdf", n) for n in (2, 3, 4)]


def test_the_pressurized_spelling_and_an_invented_quote_against_the_real_manuals(real_index):
    spelled = LIVE[1].replace("pressurised", "pressurized")
    fake = line("corrosion_control_guide.txt", 3, "Always replace the whole pipe network immediately without any inspection.")
    a, b = c.verify_citations([spelled, fake], real_index)
    assert a["status"] == "corrected" and "pressurised" in a["quote"] and "pressurized" in a["agent_quote"]
    assert b["status"] == "unverified"
