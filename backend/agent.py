"""Stateless diagnosis agent.

The backend (app/service.py) already owns the human approve / reject step and the case
status, so this agent does NOT wait for a human. It takes one defect report and returns a
diagnosis in exactly the shape the backend's DiagnosisIn model expects.

    receive_defect -> retrieve_context -> (low_confidence | reason) -> finalize

Use from code:   from agent import diagnose;  diagnose({"defect_type": ..., "location": ..., "confidence": ...})
Try it:          python agent.py
"""
import json
import re
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
import core
from core import call_llm, call_llm_backup, retrieve, FAKE_DEFECTS

LOW_CONFIDENCE_THRESHOLD = 0.70
VALID_SEVERITIES = ("low", "medium", "high", "critical")


class NoRelevantSOPError(Exception):
    """No SOP excerpt was found for this defect. The backend requires a manual + page citation
    for every diagnosis, so we refuse to make one up; the API reports this as a controlled error."""


class AgentState(TypedDict, total=False):
    defect_report: dict
    rejection_feedback: Optional[str]
    retrieved_docs: list
    needs_review: bool
    llm_output: str
    grounding: dict
    diagnosis: dict


# ---------------- Helpers ----------------

def _extract_json(text: str) -> dict:
    """Pull the first {...} object out of an LLM reply. Returns {} if there isn't a valid one."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return {}
    try:
        data = json.loads(text[start:end + 1])
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def _clamp_confidence(value) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def _quote_in_source(quote, source_text: str) -> bool:
    """True if the model's quoted evidence really appears in the retrieved excerpt (ignores case/punctuation).
    Too-short quotes (< 4 words) don't count: they would match almost anything."""
    q = _norm(quote or "")
    return len(q.split()) >= 4 and q in _norm(source_text)


# ---------------- Nodes ----------------

def _detector_notes(defect: dict) -> str:
    """Text appended to a normal (automatic) explanation: the CNN's low-trust dissent and any detector warnings."""
    notes = ""
    dissent = defect.get("cnn_dissent")
    if dissent:
        notes += (f" Note (CNN dissent, low trust): the CNN said {dissent['label']} (confidence {dissent['confidence']}) "
                  f"but the vision models agree on {defect['defect_type']}; the CNN was trained on drawings and is only "
                  f"advisory.")
    for w in defect.get("warnings") or []:
        notes += f" Warning: {w}"
    return notes


def receive_defect(state: AgentState) -> dict:
    return {"needs_review": False}


def retrieve_context(state: AgentState) -> dict:
    defect = state["defect_report"]
    query = f"{defect['defect_type']} at {defect['location']}"
    docs = retrieve(query)
    if not docs:
        # Stop BEFORE the LLM is called: without a source there is nothing to cite.
        raise NoRelevantSOPError(f"No SOP found for '{query}'")
    return {"retrieved_docs": docs}


def _detector_parts(d: dict) -> list:
    parts = []
    for key, who in (("cnn", "CNN"), ("vision_model", None), ("vision_model_2", None)):
        x = d.get(key)
        if x is None:
            continue
        name = who or x.get("name") or "vision model"
        parts.append(f"{name} says {x['label']} (confidence {x.get('confidence')})" if x.get("label")
                     else f"{name} gave no answer")
    return parts


def _candidates(defect: dict) -> list:
    """[(label, [detector names])] for the labels the detectors proposed, vision models first. Empty without detector details."""
    d = defect.get("detectors") or {}
    out: dict = {}
    for key, who in (("vision_model", None), ("vision_model_2", None), ("cnn", "CNN")):
        x = d.get(key)
        if x and x.get("label"):
            out.setdefault(x["label"], []).append(who or str(x.get("name") or "vision model").split("/")[-1])
    return list(out.items())


def _confirm_action(defect: dict, ref: str) -> str:
    """The recommended action for a disputed case: a person confirms WHICH defect it is, naming every candidate. It never
    proposes repair steps for any of them (a repair step for the wrong candidate could do harm)."""
    cands = _candidates(defect)
    if not cands:
        return (f"Manually verify the {defect['defect_type']} at {defect['location']}{ref} "
                f"before taking any maintenance action.")
    listed = "; ".join(f"{label.replace('_', ' ')} (proposed by {', '.join(names)})" for label, names in cands)
    return (f"A person must confirm which defect this is before any maintenance action. Candidates: {listed}. "
            f"Do not act on any of them until one is confirmed by looking at the photo.")


def _review_reason(defect: dict) -> str:
    d = defect.get("detectors") or {}
    parts = _detector_parts(d)
    votes = f" Votes: {d['suggestion']}." if d.get("suggestion") else ""
    warn = "".join(f" Warning: {w}" for w in (d.get("warnings") or []))
    tie = d.get("cnn_tie")
    if tie:
        return ("The two vision models DISAGREE: " + "; ".join(p for p in parts if not p.startswith("CNN")) + ". The CNN matches one "
                f"of them but was not confident enough to break the tie (CNN says {tie['label']} with confidence "
                f"{tie['confidence']}, below its {tie['threshold']} cut-off), so a person must decide." + votes + warn)
    if str(d.get("decision") or "").startswith("both vision models failed"):
        return ("BOTH VISION MODELS FAILED, so only the CNN answered (low trust: it was trained on drawings and is weak "
                "on real photos): " + "; ".join(parts) + "." + warn)
    return ("The detectors DISAGREE: " + "; ".join(parts) + "." + votes + warn +
            " The confidences are from different models and are not combined.")


def low_confidence(state: AgentState) -> dict:
    """Detection itself is doubtful: skip the LLM and ask a human to verify the detection first.

    The backend's severity field has no 'unverified' value, so severity is a PLACEHOLDER ("medium").
    That is made explicit in two places: the explanation starts with "UNVERIFIED", and the stored raw
    JSON has needs_manual_verification=True and severity_verified=False."""
    defect = state["defect_report"]
    top = state["retrieved_docs"][0]
    ref = f" (image {defect['image_ref']})" if defect.get("image_ref") else ""
    if defect.get("needs_human_review"):
        why = _review_reason(defect)
    else:
        why = (f"Detection confidence {defect['confidence']} is below the "
               f"{LOW_CONFIDENCE_THRESHOLD} threshold, so this may be a false positive.")
    return {
        "needs_review": True,
        "diagnosis": {
            "defect": f"{defect['defect_type']} at {defect['location']}",
            "severity": "medium",
            "explanation": f"UNVERIFIED DETECTION (severity is a placeholder, not assessed). {why}",
            "recommended_action": (_confirm_action(defect, ref) if defect.get("needs_human_review") else
                                   f"Manually verify the {defect['defect_type']} at {defect['location']}{ref} "
                                   f"before taking any maintenance action."),
            "source_manual": top["manual"],
            "source_page": top["page"],
        },
    }


def reason(state: AgentState) -> dict:
    defect = state["defect_report"]
    docs = state["retrieved_docs"]

    excerpts = "\n".join(
        f"[{i}] ({d['manual']}, page {d['page']}) {d['text']}" for i, d in enumerate(docs)
    )
    feedback_block = ""
    if state.get("rejection_feedback"):
        feedback_block = (
            "\nA human reviewer rejected the previous proposal with this feedback:\n"
            f"{state['rejection_feedback']}\nPropose a different action that addresses it.\n"
        )

    prompt = f"""A defect was detected:
Type: {defect['defect_type']}
Location: {defect['location']}
Confidence: {defect['confidence']}

Relevant SOP excerpts (numbered):
{excerpts}
{feedback_block}
Based ONLY on these excerpts, explain what this defect means and propose one specific action
(e.g. "schedule inspection", "shut down line", "log and monitor").

Reply with ONLY a JSON object, no other text:
{{"explanation": "...", "recommended_action": "...", "severity": "low|medium|high|critical", "source_index": <number of the excerpt you relied on>, "evidence_quote": "<5-25 words copied EXACTLY from that excerpt that support your action>"}}"""

    if getattr(core, "offline_writer_active", lambda: False)():
        return _offline_recommendation(defect, docs)  # no LLM key at all: copy the manual's own action step (labelled)

    primary = _attempt(call_llm, core.primary_text_model(), "primary", prompt, docs, retry_invalid_json=True)
    attempts = [primary]
    winner = primary if primary["ok"] else None
    if winner is None:  # the backup text model gets ONE try, never a loop
        backup = _attempt(call_llm_backup, core.text_fallback_model(), "fallback", prompt, docs, retry_invalid_json=False)
        attempts.append(backup)
        winner = backup if backup["ok"] else None
    trail = [{"role": a["role"], "model": a["model"], "problem": a["problem"]} for a in attempts]

    if winner is None:
        return _recommendation_unavailable(defect, docs, attempts, trail)

    data, source, quote = winner["data"], winner["source"], winner["quote"]
    severity = str(data.get("severity", "")).strip().lower()
    if severity not in VALID_SEVERITIES:
        severity = "medium"
    written_by = f"written by {winner['model']}" + (" (fallback)" if winner["role"] == "fallback" else "")
    return {
        "llm_output": winner["output"],
        "grounding": {
            "evidence_quote": quote or None,
            "evidence_quote_verified": True,  # a recommendation whose quote is not in the cited chunk is never accepted
            "json_retried": winner["retried"],
            "json_valid": True,
            "writer": {"model": winner["model"], "role": winner["role"], "written_by": written_by, "attempts": trail},
        },
        "diagnosis": {
            "defect": f"{defect['defect_type']} at {defect['location']}",
            "severity": severity,
            "explanation": (str(data.get("explanation") or winner["output"] or "No explanation returned by the model.").strip()
                            + _detector_notes(defect) + f" ({written_by})"),
            "recommended_action": str(data.get("recommended_action") or "Schedule a manual inspection.").strip(),
            "source_manual": source["manual"],
            "source_page": source["page"],
        },
    }


_ACTION = re.compile(r"Action Required:\s*(.+?)(?:\s*Priority:\s*(\w+)\.?)?\s*$", re.S)
_PRIORITY_TO_SEVERITY = {"low": "low", "medium": "medium", "high": "high", "critical": "critical"}


def _offline_recommendation(defect: dict, docs: list) -> dict:
    """No LLM is configured (so the app can run with no keys): the recommendation is the 'Action Required' step copied from the
    first retrieved manual passage that has one. It is labelled as a template, not an AI answer, and the quote is verified by
    construction. If no passage has an action step, the case is flagged for a person."""
    source, action, priority = docs[0], None, None
    if re.sub(r"[^a-z]", "", str(defect["defect_type"]).lower()) in ("normal", "nodefect"):
        action, priority = "No defect was detected: continue the routine inspection rounds and look again if anything seems off.", "low"
        docs = []  # a repair step copied from a nearby manual page would be misleading for a part that looks fine
    for d in docs:
        m = _ACTION.search(d["text"])
        if m:
            source, action, priority = d, " ".join(m.group(1).split()), (m.group(2) or "").lower()
            break
    quote = " ".join(action.split()[:25]) if action and docs else ""
    written_by = "written by an offline template from the manual (no LLM is configured)"
    if action and not docs:
        explanation = ("Template text, not written by an AI model (no LLM key is configured): the detector reports no defect, "
                       "so no repair step is proposed.")
    elif action:
        explanation = (f"Template text, not written by an AI model (no LLM key is configured): the detector reports "
                       f"{defect['defect_type']}, and the recommended action below is the 'Action Required' step copied from "
                       f"{source['manual']}, page {source['page']}.")
    else:
        explanation = (f"Template text, not written by an AI model (no LLM key is configured): the detector reports "
                       f"{defect['defect_type']}, but the retrieved manual pages have no 'Action Required' step, so a person "
                       f"must read {source['manual']}, page {source['page']}.")
    return {
        "llm_output": None,
        **({} if action else {"needs_review": True}),
        "grounding": {"evidence_quote": quote or None, "evidence_quote_verified": bool(quote) and _quote_in_source(quote, source["text"]),
                      "json_retried": False, "json_valid": True,
                      "writer": {"model": None, "role": "offline", "written_by": written_by, "attempts": []}},
        "diagnosis": {
            "defect": f"{defect['defect_type']} at {defect['location']}",
            "severity": _PRIORITY_TO_SEVERITY.get(priority or "", "medium"),
            "explanation": explanation + _detector_notes(defect) + f" ({written_by})",
            "recommended_action": action or f"Review the cited manual page ({source['manual']}, page {source['page']}) and decide the maintenance action.",
            "source_manual": source["manual"],
            "source_page": source["page"],
        },
    }


def _attempt(fn, model_name: str, role: str, prompt: str, docs: list, retry_invalid_json: bool) -> dict:
    """One writer attempt. ok=True only for a non-empty, valid-JSON recommendation whose evidence quote really
    appears in the cited manual chunk (our existing citation check). Otherwise ok=False and `problem` says why."""
    core.LAST_LLM.update(provider=None, model=None)  # so the model name below is this call's, not a stale one
    res = {"ok": False, "role": role, "model": model_name, "problem": None, "retried": False}
    try:
        output = fn(prompt)
        data = _extract_json(output)
        if not data and retry_invalid_json:  # model ignored the JSON format: ask once more before giving up on it
            res["retried"] = True
            output = fn(prompt + "\n\nYour last reply was not valid JSON. Reply with ONLY the JSON object.")
            data = _extract_json(output)
    except Exception as exc:  # timeout, HTTP 5xx / 429 after the transport retries, no provider configured ...
        res["problem"] = f"call failed ({type(exc).__name__})"
        return res
    res["model"] = (core.LAST_LLM.get("model") or model_name).split("/")[-1]
    res["output"] = output
    if not (output or "").strip():
        res["problem"] = "empty response"
        return res
    if not data:
        res["problem"] = "invalid JSON"
        return res
    try:
        idx = int(data.get("source_index", 0))
    except (TypeError, ValueError):
        idx = 0
    source = docs[idx] if 0 <= idx < len(docs) else docs[0]
    quote = str(data.get("evidence_quote") or "").strip()
    if not _quote_in_source(quote, source["text"]):
        res["problem"] = "citation check failed (the evidence quote is not in the cited manual text)"
        res["unverified"] = {"quote": quote or None}
        return res
    res.update(ok=True, data=data, source=source, quote=quote)
    return res


def _recommendation_unavailable(defect: dict, docs: list, attempts: list, trail: list) -> dict:
    """Both text models failed. No recommendation is invented: the case still shows every detector's result and the
    manual citation, severity is a placeholder, and a human must decide. Never raises."""
    top = docs[0]
    why = "; ".join(f"{a['role']} {a['model']}: {a['problem']}" for a in attempts)
    detectors = _detector_parts(defect.get("detectors") or {})
    seen = f" Detector results: {'; '.join(detectors)}." if detectors else ""
    ref = f" (image {defect['image_ref']})" if defect.get("image_ref") else ""
    return {
        "llm_output": None,
        "needs_review": True,
        "grounding": {"evidence_quote": None, "evidence_quote_verified": False, "json_retried": attempts[0]["retried"],
                      "json_valid": False,
                      "writer": {"model": None, "role": "none", "written_by": "no text model produced an answer",
                                 "attempts": trail}},
        "diagnosis": {
            "defect": f"{defect['defect_type']} at {defect['location']}",
            "severity": "medium",
            "explanation": ("RECOMMENDATION UNAVAILABLE (severity is a placeholder, not assessed). The AI writer could not "
                            f"produce a verified recommendation ({why}).{seen}" + _detector_notes(defect)),
            "recommended_action": (f"Recommendation unavailable. Manually review the {defect['defect_type']} at "
                                   f"{defect['location']}{ref} with the cited manual page before any maintenance action."),
            "source_manual": top["manual"],
            "source_page": top["page"],
        },
    }


def finalize(state: AgentState) -> dict:
    defect = state["defect_report"]
    diagnosis = dict(state["diagnosis"])
    diagnosis["confidence"] = _clamp_confidence(defect.get("confidence"))
    diagnosis["raw"] = {
        "defect_report": defect,
        "needs_manual_verification": state.get("needs_review", False),
        "severity_verified": not state.get("needs_review", False),
        "retrieved": [{"manual": d["manual"], "page": d["page"]} for d in state["retrieved_docs"]],
        "rejection_feedback": state.get("rejection_feedback"),
        "llm_output": state.get("llm_output"),
        "grounding": state.get("grounding"),
        # which model actually answered (None when the LLM was skipped, e.g. low confidence)
        "llm": dict(getattr(core, "LAST_LLM", None) or {}) or None if state.get("llm_output") is not None else None,
    }
    return {"diagnosis": diagnosis}


# ---------------- Routing ----------------

def route_after_retrieve(state: AgentState) -> str:
    report = state["defect_report"]
    # dual detector: when the CNN and the vision model disagree, a human verifies before any action
    if report.get("needs_human_review") or report["confidence"] < LOW_CONFIDENCE_THRESHOLD:
        return "low_confidence"
    return "reason"


# ---------------- Graph wiring ----------------

graph = StateGraph(AgentState)
graph.add_node("receive_defect", receive_defect)
graph.add_node("retrieve_context", retrieve_context)
graph.add_node("low_confidence", low_confidence)
graph.add_node("reason", reason)
graph.add_node("finalize", finalize)

graph.set_entry_point("receive_defect")
graph.add_edge("receive_defect", "retrieve_context")
graph.add_conditional_edges("retrieve_context", route_after_retrieve, {
    "low_confidence": "low_confidence",
    "reason": "reason",
})
graph.add_edge("low_confidence", "finalize")
graph.add_edge("reason", "finalize")
graph.add_edge("finalize", END)

app = graph.compile()


# ---------------- Public entry point ----------------

def diagnose(defect_report: dict, rejection_feedback: Optional[str] = None) -> dict:
    """defect_report needs: defect_type, location, confidence (0-1); image_ref is optional.
    Returns a dict that fits the backend's DiagnosisIn model."""
    result = app.invoke({"defect_report": defect_report, "rejection_feedback": rejection_feedback})
    return result["diagnosis"]


# ---------------- Run on the fake defects ----------------

if __name__ == "__main__":
    for defect in FAKE_DEFECTS:
        print(f"\n=== {defect['defect_type']} @ {defect['location']} (conf {defect['confidence']}) ===")
        out = diagnose(defect)
        out.pop("raw", None)
        print(json.dumps(out, indent=2))