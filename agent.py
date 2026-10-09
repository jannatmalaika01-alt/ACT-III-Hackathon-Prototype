import sys
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from core import call_llm, retrieve, FAKE_DEFECTS

LOW_CONFIDENCE_THRESHOLD = 0.70
MAX_ATTEMPTS = 3


class AgentState(TypedDict, total=False):
    defect_report: dict
    retrieved_docs: list
    reasoning: str
    proposed_action: str
    needs_review: bool
    attempts: int
    approval_status: str
    rejection_feedback: Optional[str]
    final_output: Optional[dict]


# ---------------- Nodes ----------------

def receive_defect(state: AgentState) -> dict:
    return {
        "attempts": 0,
        "needs_review": False,
        "approval_status": "pending",
        "rejection_feedback": None,
    }


def retrieve_context(state: AgentState) -> dict:
    defect = state["defect_report"]
    query = f"{defect['defect_type']} at {defect['location']}"
    return {"retrieved_docs": retrieve(query)}


def low_confidence(state: AgentState) -> dict:
    """Skip the LLM: detection itself is doubtful, so a human verifies first."""
    defect = state["defect_report"]
    return {
        "needs_review": True,
        "reasoning": (
            f"Detection confidence {defect['confidence']} is below the "
            f"{LOW_CONFIDENCE_THRESHOLD} threshold, so the detection may be a false positive."
        ),
        "proposed_action": (
            f"Manually verify the {defect['defect_type']} at {defect['location']} "
            f"(image {defect['image_ref']}) before taking any maintenance action."
        ),
    }


def reason(state: AgentState) -> dict:
    context = "\n".join(state["retrieved_docs"])
    defect = state["defect_report"]

    feedback_block = ""
    if state.get("rejection_feedback"):
        feedback_block = (
            f"\nA human reviewer rejected your previous proposal with this feedback:\n"
            f"{state['rejection_feedback']}\nPropose a different action that addresses it.\n"
        )

    prompt = f"""A defect was detected:
Type: {defect['defect_type']}
Location: {defect['location']}
Confidence: {defect['confidence']}

Relevant SOP excerpts:
{context}
{feedback_block}
Based on the SOP, explain what this defect means and propose one specific action
(e.g. "schedule inspection", "shut down line", "log and monitor"). Respond in two parts:
REASONING: ...
ACTION: ..."""

    output = call_llm(prompt)
    action = output.split("ACTION:")[-1].strip() if "ACTION:" in output else output
    return {
        "reasoning": output,
        "proposed_action": action,
        "attempts": state.get("attempts", 0) + 1,
    }


def human_approval(state: AgentState) -> dict:
    """Pauses the graph until someone resumes it with {"approved": bool, "feedback": str}."""
    decision = interrupt({
        "defect": state["defect_report"],
        "proposed_action": state["proposed_action"],
        "reasoning": state["reasoning"],
        "needs_review": state.get("needs_review", False),
        "attempt": state.get("attempts", 0),
    })
    if decision.get("approved"):
        return {"approval_status": "approved", "rejection_feedback": None}
    return {
        "approval_status": "rejected",
        "rejection_feedback": decision.get("feedback", "") or "No reason given.",
    }


def finalize(state: AgentState) -> dict:
    status = state["approval_status"]
    if status == "rejected":
        status = "escalated_to_human"  # rejected too many times, hand off to a person
    return {
        "final_output": {
            "status": status,
            "action": state["proposed_action"],
            "reasoning": state["reasoning"],
            "low_confidence": state.get("needs_review", False),
            "attempts": state.get("attempts", 0),
        }
    }


# ---------------- Routing ----------------

def route_after_retrieve(state: AgentState) -> str:
    if state["defect_report"]["confidence"] < LOW_CONFIDENCE_THRESHOLD:
        return "low_confidence"
    return "reason"


def route_after_approval(state: AgentState) -> str:
    if state["approval_status"] == "approved":
        return "finalize"
    if state.get("attempts", 0) >= MAX_ATTEMPTS:
        return "finalize"
    return "reason"


# ---------------- Graph wiring ----------------

graph = StateGraph(AgentState)
graph.add_node("receive_defect", receive_defect)
graph.add_node("retrieve_context", retrieve_context)
graph.add_node("low_confidence", low_confidence)
graph.add_node("reason", reason)
graph.add_node("human_approval", human_approval)
graph.add_node("finalize", finalize)

graph.set_entry_point("receive_defect")
graph.add_edge("receive_defect", "retrieve_context")
graph.add_conditional_edges("retrieve_context", route_after_retrieve, {
    "low_confidence": "low_confidence",
    "reason": "reason",
})
graph.add_edge("low_confidence", "human_approval")
graph.add_edge("reason", "human_approval")
graph.add_conditional_edges("human_approval", route_after_approval, {
    "finalize": "finalize",
    "reason": "reason",
})
graph.add_edge("finalize", END)

# MemorySaver is required for interrupt(): it stores the paused state per thread_id
app = graph.compile(checkpointer=MemorySaver())


# ---------------- Run (terminal stand-in for the frontend) ----------------

def run_defect(defect: dict, thread_id: str, auto: bool = False) -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    app.invoke({"defect_report": defect}, config)

    # While the graph is paused at human_approval, ask for a decision and resume it
    while app.get_state(config).next:
        pending = app.get_state(config).tasks[0].interrupts[0].value

        print(f"\n[Approval needed] attempt {pending['attempt']}"
              + ("  (LOW CONFIDENCE)" if pending["needs_review"] else ""))
        print(f"Proposed action: {pending['proposed_action']}")

        if auto:
            decision = {"approved": True, "feedback": ""}
        else:
            answer = input("Approve? (y = approve, n = reject): ").strip().lower()
            if answer == "y":
                decision = {"approved": True, "feedback": ""}
            else:
                feedback = input("Why reject? (this goes back to the agent): ")
                decision = {"approved": False, "feedback": feedback}

        app.invoke(Command(resume=decision), config)

    return app.get_state(config).values["final_output"]


if __name__ == "__main__":
    auto = "--auto" in sys.argv  # python agent.py --auto  -> approves everything
    for i, defect in enumerate(FAKE_DEFECTS):
        print(f"\n=== {defect['defect_type']} @ {defect['location']} (conf {defect['confidence']}) ===")
        result = run_defect(defect, thread_id=f"defect-{i}", auto=auto)
        print("FINAL:", result)
