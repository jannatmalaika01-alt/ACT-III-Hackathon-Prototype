"""Print the shared-contract JSON for any file.   python run_pipeline.py [path] [--case-id C-101]

With no path it runs the sample image data/test/test_02.png.
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from knowledge.pipeline import VisionKnowledgePipeline  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default=os.path.join(ROOT, "data", "test", "test_02.png"))
    ap.add_argument("--case-id", default="C-101")
    a = ap.parse_args()
    print(json.dumps(VisionKnowledgePipeline().process_case(a.path, case_id=a.case_id), indent=2))
