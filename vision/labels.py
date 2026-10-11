"""Single source of truth for defect labels.

Internal ids use underscores (file names, CSV, model classes). The shared team contract uses the
readable form ("bearing wear"), so everything leaving this package goes through `to_contract_label`.
"""
CLASSES = ["normal", "bearing_wear", "structural_crack", "corrosion"]
NORMAL = "normal"


def to_contract_label(class_id: str) -> str:
    return class_id.replace("_", " ")


def from_contract_label(label: str) -> str:
    return label.strip().lower().replace(" ", "_")
