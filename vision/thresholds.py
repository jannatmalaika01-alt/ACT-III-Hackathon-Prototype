"""The one place the CNN's "a person should verify this" cut-off lives.

vision/defect_detector.py uses it to word its explanation, and the backend's dual detector (DUAL_MODE=vision_primary) uses it
to decide whether the CNN is confident enough to break a tie between the two vision models. Change it here only.
"""
LOW_CONFIDENCE = 0.70
