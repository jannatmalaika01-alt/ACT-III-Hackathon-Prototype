"""The vision prompt and the tolerant answer parser (retry once, then a clean typed failure)."""
import pytest
from PIL import Image

from app import dual_detector as dd


# ---------------- prompt ----------------

def test_prompt_defines_every_class_explicitly():
    for text in ("bearing_wear: any bearing, roller, race or bearing housing that shows wear, pitting, scoring, "
                 "spalling, discoloration or rust",
                 "corrosion: rust or oxidation on plates, bolts, pipes, frames, tanks or other non-bearing metal",
                 "structural_crack: cracks or spalling in walls, concrete, welds, castings or brackets",
                 "normal: clean, undamaged part with no visible defect",
                 'Reply with ONLY JSON: {"defect_type": "<label>", "confidence": <number between 0 and 1>}'):
        assert text in dd.PROMPT


# ---------------- parser ----------------

@pytest.mark.parametrize("text,label,conf", [
    ('{"defect_type": "corrosion", "confidence": 0.93}', "corrosion", 0.93),
    ('```json\n{"defect_type": "Bearing Wear", "confidence": 0.8}\n```', "bearing_wear", 0.8),
    ('Sure! {"defect_type": "structural-crack", "confidence": "85%"}', "structural_crack", 0.85),
    ('{"label": "normal.", "confidence": 90}', "normal", 0.9),
    ('{"defect_type": "rust", "confidence": 0.7}', "corrosion", 0.7),               # alias
    ('{"defect_type": "crack"}', "structural_crack", None),                         # no confidence
    ('{"defect_type": "corrosion", "confidence": "high"}', "corrosion", None),      # unusable confidence
    ('defect_type: corrosion, confidence: 0.6', "corrosion", 0.6),                  # not JSON at all
    ('{"defect_type": "corrosion", "confidence": 0.9', "corrosion", 0.9),           # truncated JSON
])
def test_tolerant_parsing(text, label, conf):
    assert dd.parse_vision_answer(text) == {"label": label, "confidence": conf}


@pytest.mark.parametrize("text", ["", "I cannot tell.", "{}", '{"defect_type": "banana"}', '{"defect_type": ""}',
                                  "[1, 2, 3]", None])
def test_unusable_answers_raise_the_typed_error_not_valueerror(text):
    with pytest.raises(dd.VisionAnswerError) as e:
        dd.parse_vision_answer(text)
    assert not isinstance(e.value, ValueError) and e.value.kind == "parse_error"


# ---------------- the call: retry once, then fail cleanly ----------------

class Reply:
    def __init__(self, content):
        self.content = content

    def json(self):
        return {"model": "Flash", "choices": [{"message": {"content": self.content}}]}


@pytest.fixture
def post(monkeypatch, tmp_path):
    import core
    monkeypatch.setenv("TOKENFACTORY_BASE_URL", "https://tf.example/v1")
    monkeypatch.setenv("TOKENFACTORY_API_KEY", "test-key-not-real")
    state = {"script": [], "payloads": []}

    def fake(p, payload):
        state["payloads"].append(payload)
        item = state["script"].pop(0)
        if isinstance(item, Exception):
            raise item
        return Reply(item)

    monkeypatch.setattr(core, "_post_with_retries", fake)
    img = tmp_path / "a.png"
    Image.new("RGB", (32, 32)).save(img)
    state["path"] = str(img)
    return state


def test_a_good_answer_is_used_directly(post):
    post["script"] = ['{"defect_type": "corrosion", "confidence": 0.9}']
    assert dd.run_vision_model(post["path"])["label"] == "corrosion"
    assert len(post["payloads"]) == 1


def test_an_unusable_answer_is_retried_once_with_the_same_prompt(post):
    post["script"] = ["I think it is rusty, maybe.", '{"defect_type": "corrosion", "confidence": 0.8}']
    r = dd.run_vision_model(post["path"])
    assert r["label"] == "corrosion" and len(post["payloads"]) == 2
    assert post["payloads"][0]["messages"] == post["payloads"][1]["messages"]  # the same prompt, nothing added


def test_an_unknown_label_is_retried_once(post):
    post["script"] = ['{"defect_type": "banana", "confidence": 1}', '{"defect_type": "normal", "confidence": 0.9}']
    assert dd.run_vision_model(post["path"])["label"] == "normal"


def test_two_bad_answers_end_in_a_clean_error_not_a_loop(post):
    post["script"] = ["nope", '{"defect_type": "banana"}']
    with pytest.raises(dd.VisionAnswerError):
        dd.run_vision_model(post["path"])
    assert len(post["payloads"]) == 2


def test_an_empty_reply_is_also_retried(post):
    post["script"] = [None, '{"defect_type": "bearing_wear", "confidence": 0.7}']
    assert dd.run_vision_model(post["path"])["label"] == "bearing_wear"


def test_a_failed_vision_call_never_crashes_the_dual_pipeline(post):
    post["script"] = ["nope", "nope"]
    r = dd.run_dual("x.png", lambda p: {"label": "corrosion", "confidence": 0.9},
                    lambda p: dd.run_vision_model(post["path"]), timeout=5, dual_mode="pair")
    assert r["agreement"] == "single_detector" and r["vision_model"]["error"] == "VisionAnswerError"


def test_voting_script_records_a_parse_error_instead_of_valueerror(post, monkeypatch):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("voting_compare", Path(__file__).parents[1] / "scripts" / "voting_compare.py")
    vc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vc)
    post["script"] = ["nope", "nope"]
    out = vc._ask(dd, "Qwen3.8-Flash-Next", post["path"])
    assert out["label"] is None and out["error"] == "parse_error"
