"""Query expansion with maintenance vocabulary, so a short query like "corrosion at pipe section B7" finds the
procedure page and not just a general overview. Matches by keyword, so the backend agent's own defect names
(crack, loose_bolt, misalignment, ...) work as well as ours."""

# (keywords that trigger it, terms appended to the query)
EXPANSIONS = [
    (("bearing",), "bearing wear fatigue scoring marks metallic noise overheating vibration replace bearing"),
    (("crack", "fracture"), "structural crack linear fracture welded support seam repair"),
    (("corros", "rust", "oxid"), "corrosion rust oxide flakes surface pitting paint degradation wall loss treatment"),
    (("bolt", "fastener", "loose"), "loose bolt re-tighten torque fastener work order"),
    (("misalign",), "misalignment conveyor belt shaft laser alignment"),
    (("normal", "no defect", "healthy"), "routine inspection rounds daily checks general overview"),
]


def expand_query(query: str) -> str:
    q = query.lower().replace("_", " ")
    extra = [terms for keys, terms in EXPANSIONS if any(k in q for k in keys)]
    return f"{query}. {' '.join(extra)}" if extra else query
