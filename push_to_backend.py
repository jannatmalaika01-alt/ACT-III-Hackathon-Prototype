"""Send one file through the whole chain against a running backend.

    python push_to_backend.py data/test/test_02.png --api http://localhost:8000 --location "motor M3 bearing housing"

1. uploads the file          POST /cases
2. detects the defect and finds manual evidence (this repo's pipeline)
3. sends the defect report   POST /cases/{id}/analyze   (falls back to /diagnose on older backends)
The backend agent then writes the cited diagnosis; the case stays 'pending' until a human approves it.
"""
import argparse
import json
import mimetypes
import os
import sys

import httpx

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from knowledge.pipeline import VisionKnowledgePipeline  # noqa: E402

CONTENT_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp", ".pdf": "application/pdf",
                 ".csv": "text/csv", ".json": "application/json", ".txt": "text/plain"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--location", default="location not specified in upload")
    ap.add_argument("--uploaded-by", default="titus")
    args = ap.parse_args()

    ctype = CONTENT_TYPES.get(os.path.splitext(args.file)[1].lower()) or mimetypes.guess_type(args.file)[0] or "application/octet-stream"
    with httpx.Client(base_url=args.api, timeout=120) as api:
        with open(args.file, "rb") as f:
            case = api.post("/cases", files={"file": (os.path.basename(args.file), f, ctype)}, data={"uploaded_by": args.uploaded_by})
        case.raise_for_status()
        case_id = case.json()["id"]

        contract = VisionKnowledgePipeline().process_case(args.file, case_id=case_id)
        report = {"defect_type": contract["defect"]["type"], "location": args.location,
                  "confidence": contract["defect"]["confidence"], "image_ref": os.path.basename(args.file)}

        r = api.post(f"/cases/{case_id}/analyze", json=report)
        if r.status_code in (404, 405):  # backend without /analyze
            r = api.post(f"/cases/{case_id}/diagnose", json=report)
        print("--- vision/knowledge contract ---")
        print(json.dumps(contract, indent=2))
        print(f"--- backend {r.status_code} ---")
        print(json.dumps(r.json(), indent=2))
        sys.exit(0 if r.status_code == 200 else 1)


if __name__ == "__main__":
    main()
