from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from core import call_llm, retrieve, FAKE_DEFECTS


class AgentState(TypedDict):
    defect_report: dict
    retrieved_docs: list
    reasoning: str
    proposed_action: str
    approval_status: str
    rejection_feedback: Optional[str]
    final_output: Optional[dict]


# ---------------- Nodes ----------------

def receive_defect(state: AgentState) -> AgentState:
    return state


def retrieve_context(state: AgentState) -> AgentState:
    defect = state["defect_report"]
    query = f"{defect['defect_type']} at {defect['location']}"
    state["retrieved_docs"] = retrieve(query)
    return state


def reason(state: AgentState) -> AgentState:
    context = "\n".join(state["retrieved_docs"])
    defect = state["defect_report"]

    prompt = f"""A defect was detected:
Type: {defect['defect_type']}
Location: {defect['location']}
Confidence: {defect['confidence']}

Relevant SOP excerpts:
{context}

Based on the SOP, explain what this defect means and propose one specific action
(e.g. "schedule inspection", "shut down line", "log and monitor"). Respond in two parts:
REASONING: ...
ACTION: ..."""

    output = call_llm(prompt)
    state["reasoning"] = output
    state["proposed_action"] = output.split("ACTION:")[-1].strip() if "ACTION:" in output else output
    return state


def human_approval(state: AgentState) -> AgentState:
    state["approval_status"] = "pending"  # TODO: wire to real approval signal later
    return state


def finalize(state: AgentState) -> AgentState:
    state["final_output"] = {
        "status": state["approval_status"],
        "action": state["proposed_action"],
        "reasoning": state["reasoning"],
    }
    return state


def route_after_approval(state: AgentState) -> str:
    return "finalize" if state["approval_status"] != "rejected" else "reason"


# ---------------- Graph wiring ----------------

graph = StateGraph(AgentState)
graph.add_node("receive_defect", receive_defect)
graph.add_node("retrieve_context", retrieve_context)
graph.add_node("reason", reason)
graph.add_node("human_approval", human_approval)
graph.add_node("finalize", finalize)

graph.set_entry_point("receive_defect")
graph.add_edge("receive_defect", "retrieve_context")
graph.add_edge("retrieve_context", "reason")
graph.add_edge("reason", "human_approval")
graph.add_conditional_edges("human_approval", route_after_approval, {
    "finalize": "finalize",
    "reason": "reason",
})
graph.add_edge("finalize", END)

app = graph.compile()


# ---------------- Run ----------------

if __name__ == "__main__":
    for defect in FAKE_DEFECTS:
        result = app.invoke({"defect_report": defect})
        print(f"\n--- {defect['defect_type']} @ {defect['location']} ---")
        print(result["final_output"])