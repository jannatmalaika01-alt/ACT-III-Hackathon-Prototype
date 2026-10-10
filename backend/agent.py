"""Stateless diagnosis agent.

The backend (app/service.py) already owns the human approve / reject step and the case
status, so this agent does NOT wait for a human. It takes one defect report and returns a
diagnosis in exactly the shape the backend's DiagnosisIn model expects.

    receive_defect -> retrieve_context -> (low_confidence | reason) -> finalize

Use from code:   from agent import diagnose;  diagnose({"defect_type": ..., "location": ..., "confidence": ...})
Try it:          python agent.py
"""
import json
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from core import call_llm, retrieve, FAKE_DEFECTS

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


# ---------------- Nodes ----------------

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


def low_confidence(state: AgentState) -> dict:
    """Detection itself is doubtful: skip the LLM and ask a human to verify the detection first.

    The backend's severity field has no 'unverified' value, so severity is a PLACEHOLDER ("medium").
    That is made explicit in two places: the explanation starts with "UNVERIFIED", and the stored raw
    JSON has needs_manual_verification=True and severity_verified=False."""
    defect = state["defect_report"]
    top = state["retrieved_docs"][0]
    ref = f" (image {defect['image_ref']})" if defect.get("image_ref") else ""
    return {
        "needs_review": True,
        "diagnosis": {
            "defect": f"{defect['defect_type']} at {defect['location']}",
            "severity": "medium",
            "explanation": (
                f"UNVERIFIED DETECTION (severity is a placeholder, not assessed). "
                f"Detection confidence {defect['confidence']} is below the "
                f"{LOW_CONFIDENCE_THRESHOLD} threshold, so this may be a false positive."
            ),
            "recommended_action": (
                f"Manually verify the {defect['defect_type']} at {defect['location']}{ref} "
                f"before taking any maintenance action."
            ),
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
{{"explanation": "...", "recommended_action": "...", "severity": "low|medium|high|critical", "source_index": <number of the excerpt you relied on>}}"""

    output = call_llm(prompt)
    data = _extract_json(output)

    # The citation comes from the retrieved documents, never from free text, so it can't be invented.
    try:
        idx = int(data.get("source_index", 0))
    except (TypeError, ValueError):
        idx = 0
    source = docs[idx] if 0 <= idx < len(docs) else docs[0]

    severity = str(data.get("severity", "")).strip().lower()
    if severity not in VALID_SEVERITIES:
        severity = "medium"

    return {
        "llm_output": output,
        "diagnosis": {
            "defect": f"{defect['defect_type']} at {defect['location']}",
            "severity": severity,
            "explanation": str(data.get("explanation") or output or "No explanation returned by the model.").strip(),
            "recommended_action": str(
                data.get("recommended_action") or "Schedule a manual inspection."
            ).strip(),
            "source_manual": source["manual"],
            "source_page": source["page"],
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
    }
    return {"diagnosis": diagnosis}


# ---------------- Routing ----------------

def route_after_retrieve(state: AgentState) -> str:
    if state["defect_report"]["confidence"] < LOW_CONFIDENCE_THRESHOLD:
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