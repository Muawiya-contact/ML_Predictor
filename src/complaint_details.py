"""Versioned, bilingual text-detail indicators for research experiments.

These are mentions, not clinical findings or a replacement triage rubric.
Negation has a separate indicator; no label, row ID or patient outcome is read.
Ambiguous durations remain unknown instead of choosing a convenient value.
"""
import re
import numpy as np

VERSION = 1
PATTERNS = {
    'mild_word': r'\b(?:mild|slight|mamuli|halka|halki|halkay)\b',
    'strong_word': r'\b(?:severe|intense|shadeed|tez|bohat|bahut)\b',
    'sudden_word': r'\b(?:sudden|suddenly|achanak)\b',
    'exertion_word': r'\b(?:exercise|exertion|exertional|climbing|stairs|seedhiyan|chalne)\b',
    'radiation_phrase': r'\b(?:radiat\w*|tak ja|tak phel|tak phail)\b',
    'breathing_phrase': r'\b(?:breath\w*|dyspn\w*|saans|sans)\b',
    'fainting_word': r'\b(?:faint\w*|syncope|behosh\w*)\b',
    'sweating_word': r'\b(?:sweat\w*|pasina|paseena)\b',
    'nausea_word': r'\b(?:nause\w*|vomit\w*|ulti)\b',
    'family_word': r'\b(?:family|familial|khandan\w*)\b',
    'history_word': r'\b(?:history|known|purani|purana|pehle)\b',
    'hypertension_word': r'\b(?:hypertension|hypertensive)\b',
    'diabetes_word': r'\b(?:diabet\w*|diabetes|sugar)\b',
    'negation_word': r'\b(?:no|not|without|denies|nahi|nahin)\b',
    'night_word': r'\b(?:night|raat|rat)\b',
    'morning_word': r'\b(?:morning|subah)\b',
}
NUMBERS = {'one': 1, 'ek': 1, 'aik': 1, 'two': 2, 'do': 2,
           'three': 3, 'teen': 3, 'four': 4, 'char': 4,
           'five': 5, 'paanch': 5, 'six': 6, 'seven': 7, 'eight': 8,
           'nine': 9, 'ten': 10}
FEATURE_NAMES = list(PATTERNS) + ['duration_known', 'log_duration_minutes', 'duration_ambiguous']


def duration_minutes(text):
    """Return (minutes or None, ambiguity) for explicit English/Roman Urdu units."""
    text = str(text).casefold()
    found = []
    for match in re.finditer(r'\b(?:half|aadhay|aadhe|adhay|adhe|aadha)\s+(?:an?\s+)?(?:hour|ghant[ae]y?)\b', text):
        found.append((match.span(), 30.0))
    pattern = r'\b(\d+(?:\.\d+)?|' + '|'.join(NUMBERS) + r'|an?)\s+(minutes?|mins?|hours?|hrs?|ghanta|ghante|ghantay|days?|din)\b'
    for match in re.finditer(pattern, text):
        if any(match.start() < end and match.end() > start for (start, end), _ in found):
            continue
        token, unit = match.groups()
        count = float(token) if token[0].isdigit() else NUMBERS.get(token, 1)
        factor = 1440 if unit in ('day', 'days', 'din') else 60 if unit.startswith(('hour', 'hr', 'ghant')) else 1
        found.append((match.span(), count * factor))
    values = set(value for _, value in found)
    if len(values) == 1:
        return values.pop(), False
    return None, len(values) > 1


def detail_matrix(texts):
    rows = []
    for value in texts:
        text = '' if value is None else str(value).casefold()
        minutes, ambiguous = duration_minutes(text)
        rows.append([float(bool(re.search(pattern, text))) for pattern in PATTERNS.values()]
                    + [float(minutes is not None), np.log1p(minutes) if minutes is not None else 0.0, float(ambiguous)])
    return np.asarray(rows, dtype=np.float64).reshape(-1, len(FEATURE_NAMES))
