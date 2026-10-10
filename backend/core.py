import os
import re
from dotenv import load_dotenv
from groq import Groq
import chromadb
from chromadb.utils import embedding_functions

# ---------------- Config ----------------

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found - check your .env file")

# Change the model without editing code: set LLM_MODEL in .env
# (e.g. LLM_MODEL=qwen/qwen3-32b  or  llama-3.1-8b-instant). Run list_models.py to see what your key can use.
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")

client = Groq(api_key=GROQ_API_KEY)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def call_llm(prompt: str, system: str = "You are a factory maintenance assistant.") -> str:
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    text = response.choices[0].message.content or ""
    # Qwen-style models print their reasoning in <think> tags; remove it so parsing works.
    return _THINK_RE.sub("", text).strip()


# ---------------- Fake test data (until real GPU/test data arrives) ----------------

FAKE_DEFECTS = [
    {"defect_type": "crack", "location": "weld joint A3", "confidence": 0.91, "image_ref": "img_014.png"},
    {"defect_type": "corrosion", "location": "pipe section B7", "confidence": 0.78, "image_ref": "img_022.png"},
    {"defect_type": "misalignment", "location": "conveyor belt C2", "confidence": 0.65, "image_ref": "img_031.png"},
    {"defect_type": "loose_bolt", "location": "panel D5", "confidence": 0.95, "image_ref": "img_040.png"},
]

# Every SOP carries a manual name + page, because the backend refuses a diagnosis without a citation.
# The page numbers here are placeholders for the fake data; replace them with the real manuals' pages.
SOP_DOCS = [
    {
        "id": "sop_weld_inspection",
        "manual": "Weld Inspection SOP",
        "page": 12,
        "text": (
            "Weld Inspection SOP: Any crack detected in a weld joint above 0.85 confidence "
            "must be flagged for immediate manual inspection by a certified welder. Do not "
            "clear equipment for operation until the joint is re-tested. Document location "
            "and crack length in the maintenance log."
        ),
    },
    {
        "id": "sop_corrosion_handling",
        "manual": "Corrosion Handling SOP",
        "page": 7,
        "text": (
            "Corrosion Handling SOP: Surface corrosion on pipe sections should be logged and "
            "scheduled for inspection within 7 days if confidence is below 0.85. If corrosion "
            "affects structural pipe sections carrying pressurized fluid, escalate to shutdown "
            "and immediate inspection regardless of confidence score."
        ),
    },
    {
        "id": "sop_mechanical_alignment",
        "manual": "Mechanical Alignment SOP",
        "page": 4,
        "text": (
            "Mechanical Alignment SOP: Conveyor belt or equipment misalignment should be logged "
            "and monitored. If misalignment confidence exceeds 0.80 or recurs more than twice in "
            "24 hours, schedule a maintenance technician visit. Loose bolts or fasteners detected "
            "with confidence above 0.90 should trigger an immediate tightening work order."
        ),
    },
]

# ---------------- RAG retrieval (CPU only, no GPU needed) ----------------

_chroma_client = chromadb.Client()
_embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
_collection = _chroma_client.get_or_create_collection("sops", embedding_function=_embed_fn)

_collection.add(
    documents=[d["text"] for d in SOP_DOCS],
    ids=[d["id"] for d in SOP_DOCS],
    metadatas=[{"manual": d["manual"], "page": d["page"]} for d in SOP_DOCS],
)


def retrieve(query: str, n_results: int = 2) -> list:
    """Returns a list of {"manual", "page", "text"} for the best-matching SOP excerpts."""
    results = _collection.query(query_texts=[query], n_results=n_results)
    return [
        {"manual": meta["manual"], "page": int(meta["page"]), "text": text}
        for text, meta in zip(results["documents"][0], results["metadatas"][0])
    ]
