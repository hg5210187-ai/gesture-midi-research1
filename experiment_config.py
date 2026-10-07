SENSITIVITY_LEVELS = {
    "low":    {"label": "低感度", "cm_per_octave": 60},
    "medium": {"label": "中感度", "cm_per_octave": 30},
    "high":   {"label": "高感度", "cm_per_octave": 15},
}

SCALE_TYPES = {
    "chromatic":   {"label": "クロマチック", "notes": list(range(12))},
    "diatonic":    {"label": "ダイアトニック", "notes": [0,2,4,5,7,9,11]},
    "pentatonic":  {"label": "ペンタトニック", "notes": [0,2,4,7,9]},
    "continuous":  {"label": "連続 (Theremin)", "notes": None},
}

BASE_OCTAVE = 4
PITCH_RANGE_OCTAVES = 2

# Approximate vertical extent visible in the webcam frame at typical desk distance.
# Used to convert normalised hand-Y into centimetres for the cm_per_octave mapping.
ASSUMED_FRAME_HEIGHT_CM = 40
