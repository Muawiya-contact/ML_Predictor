"""
generate_cardiac_dataset.py
=======================================================================
SYNTHETIC cardiac triage dataset generator.
=======================================================================

WHY THIS EXISTS
---------------
cardiac_multilingual_10000.csv arrived with no generator and two defects
that make it unusable as published evidence:

  * ZERO Level-4 rows, so the non-urgent tier was unreachable by
    construction and the deployed model could only ever emit 3 classes.
  * 120 distinct words across all 10,000 complaints, essentially one
    sentence skeleton with slots. On text that uniform, "same-meaning
    complaints cluster together" is a property of the generator, not of
    the NLP pipeline, so Contribution 1 and 2 cannot be demonstrated.

This script replaces that process with a reviewable one.

EVERY PHRASE HERE IS DERIVED FROM REAL DATA
-------------------------------------------
The vocabulary and phrasings are taken from the cardiac subset of
triage_mixed_language_dataset_10000_RECOVERED.csv (7,855 rows, 887
distinct complaints, 636-word vocabulary) - the organic dataset this
project started from. Nothing here is invented Roman Urdu; the fragments
are recombined with varied connective grammar, not written from nothing.
The 22 genuine Level-4 complaints in that file seed the Level-4 branch.

HOW THE LABEL IS DECIDED  (read this before quoting any accuracy number)
------------------------------------------------------------------------
The triage level is sampled FIRST, then the complaint text, the ECG
status and the vitals are all sampled conditional on it, with deliberate
overlap between adjacent levels.

That is a change from the file this replaces, where ECG_Status
determined the label almost perfectly (ST elevation -> Level 1 in 1930
of 1932 rows). A model trained on that data scored 99.1% from vitals
alone while the text contributed nothing - the headline accuracy was an
ECG lookup wearing an NLP pipeline as a hat. Overlapping the
distributions means text, ECG and vitals each carry partial signal and
none of them is a giveaway.

Pass --ecg-determinism strict to reproduce the old near-deterministic
behaviour instead. It is not the default, and a dataset built that way
should not be used to support a claim about the text pipeline.

THIS DATA IS SYNTHETIC
----------------------
It is not patient data and must never be described as such. Every run
writes <output>.provenance.json recording the method, seed, generator
version and date; train_embedding_pipeline.py copies that into
model_manifest.json so the disclosure travels with the model bundle.

USAGE
-----
    python generate_cardiac_dataset.py --preview 20      # inspect first
    python generate_cardiac_dataset.py                   # write 10,000
    python generate_cardiac_dataset.py --rows 5000 --seed 7
=======================================================================
"""

import argparse
import json
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

GENERATOR_VERSION = "1.0"
DEFAULT_OUT = "cardiac_multilingual_10000.csv"
DEFAULT_ROWS = 10000
DEFAULT_SEED = 20260816

# ======================================================================
# PHRASE BANK
# Fragments lifted from the recovered cardiac complaints, grouped by the
# severity they actually appeared at. Roman Urdu spelling is kept as the
# nurses wrote it, including the inconsistencies - normalising it here
# would hide exactly the variation the pipeline exists to handle.
# ======================================================================

# --- core presenting symptom, by severity tier ---
#
# FOUR tiers, one per triage level. An earlier draft gave Levels 2 and 3
# the same "moderate" bank, which had two consequences: the two tiers were
# indistinguishable FROM THE TEXT (so the complaint could not possibly help
# separate them, which defeats the point of the dataset), and they competed
# for the same finite pool of strings - the generator ran out and refused
# to emit, which is how the problem surfaced.
SYMPTOM = {
    "severe": [
        "seena mein shadeed dard", "seene mein bohat tez dard",
        "chest mein severe pain", "seena mein dard bardasht se bahar",
        "tez seena dard", "seene mein jakran aur dard",
        "chest crushing pain", "seena mein bhaari pan aur dard",
        "dil ki dhadkan bohat tez", "saans lena mushkil",
        "saans foolna", "behoshi", "behoshi jaisa lagna",
        "seena phatne jaisa dard", "chest mein bohat zyada dabao",
        "dard jo bardasht nahi ho raha", "seena jakar gaya hai",
        "saans bilkul nahi aa rahi", "dam ghutne jaisa lag raha",
    ],
    "moderate": [
        "seena mein dard", "seene mein tez dard", "chest pain",
        "seena bhaari lag raha", "chest tightness", "seena mein jakran",
        "seena mein dabao", "dhadkan tez", "heart beat irregular",
        "dil ki dhadkan bay-tarteeb", "saans phoolna",
        "seena mein jalan", "chest mein burning",
        "seene mein kasav", "dil tez dharak raha", "chest mein pressure",
        "saans chadh rahi hai", "seena mein khichao",
    ],
    "mild_moderate": [
        "seena mein rukk rukk kar dard", "seena mein halka dabao",
        "chest mein halki tightness", "seena mein beech beech mein dard",
        "dhadkan kabhi tez kabhi normal", "seena bhaari sa lagta hai",
        "chest mein halka sa pressure", "saans thori si phoolti hai",
        "seena mein sustee wala dard", "chest mein halki jalan",
        "dil ki dhadkan zara tez", "seena mein thora sa khichao",
        "chest discomfort on and off", "seena mein halki takleef",
    ],
    "mild": [
        "seena mein halka dard", "chest mein halka dard",
        "seena mein chubhan", "seena mein halki jalan",
        "chest mein mild discomfort", "seena mein takleef",
        "dil ghabrana", "halki si dhadkan tez",
        "seena mein occasional pain", "seena mein kabhi kabhar dard",
        "seena mein zara si jalan", "chest mein halka sa khichao",
        "seena mein maamuli dard", "dil thora sa tez dharakta hai",
    ],
}

# --- associated / red-flag features ---
ASSOCIATED = {
    "severe": [
        "pasina bohat aa raha hai", "thanda pasina aa raha hai",
        "ulti bhi ho rahi hai", "chehra sun ho gaya hai",
        "baayein bazu mein bhi dard ja raha hai",
        "jaw mein bhi dard hai", "kandhay tak dard ja raha hai",
        "peeth mein dard phail raha hai", "bohat kamzori hai",
        "aankhon ke aage andhera aa raha hai", "girne wala mehsoos hota hai",
    ],
    "moderate": [
        "thakan bohat hai", "kamzori mehsoos ho rahi hai",
        "chakkar bhi aa rahe hain", "saans lene mein takleef hai",
        "left hand mein bhi dard hai", "pair mein sujan hai",
        "ghabrahat ho rahi hai", "neend nahi aa rahi",
        "halka pasina bhi aa raha hai", "bhook nahi lag rahi",
        "gardan mein bhi khichao hai", "thakawat bohat jaldi hoti hai",
        "haath thanday ho rahe hain",
    ],
    "mild_moderate": [
        "thori si thakan hai", "halki kamzori lagti hai",
        "kabhi kabhar chakkar aa jate hain", "zyada chalne par saans phoolti hai",
        "raat ko neend thori kharab hoti hai", "kaam karne par barh jata hai",
        "aaram se kam ho jata hai", "koi aur takleef nahi",
    ],
    "mild": [
        "aur koi masla nahi", "baqi sab theek hai",
        "rest se theek ho jata hai", "aaram karne se kam ho jata hai",
        "zyada takleef nahi hai", "kaam karte hue masla nahi hota",
        "roz ke kaam mein koi rukawat nahi", "khud hi theek ho jata hai",
        "dawa se aaram aa jata hai", "neend theek aa rahi hai",
    ],
}

# --- what brought it on ---
TRIGGER = {
    "severe": [
        "aaram karte hue bhi", "sotay waqt achanak", "achanak se",
        "raat ko sotay hue", "bina kisi wajah ke",
    ],
    "moderate": [
        "seedhiyan chadhte waqt", "chalte waqt", "kaam karte waqt",
        "exercise ke baad", "khana khane ke baad", "subah uthte hi",
        "tez chalne par", "seedhiyan utarte waqt", "bhaari cheez uthate waqt",
        "ghar ka kaam karte waqt",
    ],
    "mild_moderate": [
        "thora chalne par", "halka kaam karte waqt", "seedhiyan chadhne par",
        "lambi walk ke baad", "din bhar ke kaam ke baad", "shaam ko",
        "garmi mein bahar nikalne par", "zyada thak jane par",
    ],
    "mild": [
        "stress ke waqt", "anxiety ke doran", "coffee peene ke baad",
        "caffeine ki wajah se", "zyada khana khane ke baad",
        "tension lene par", "kabhi kabhar",
    ],
}

# --- finite verb phrases, for skeletons that need a clause not an adverb ---
TRIGGER_VERB = [
    "chalta hoon", "seedhiyan chadhta hoon", "kaam karta hoon",
    "exercise karta hoon", "letta hoon", "khana khata hoon",
    "tension leta hoon", "tez chalta hoon", "seedhiyan chadhti hoon",
    "kaam karti hoon", "zyada chalti hoon",
]

# --- noun-phrase symptoms only ---
# Skeletons that append their own verb ("... hota hai") cannot take a
# symptom that already ends in a participle, or you get
# "seena bhaari lag raha hota hai". These are the verb-free forms.
SYMPTOM_NP = {
    "severe": [
        "seena mein shadeed dard", "seene mein bohat tez dard",
        "chest mein severe pain", "tez seena dard", "chest crushing pain",
        "seena mein jakran aur dard", "behoshi jaisa ehsaas",
    ],
    "moderate": [
        "seena mein dard", "seene mein tez dard", "chest pain",
        "chest tightness", "seena mein jakran", "seena mein dabao",
        "seena mein jalan", "chest mein burning", "dhadkan tez",
    ],
    "mild_moderate": [
        "seena mein halka dabao", "chest mein halki tightness",
        "seena mein rukk rukk kar dard", "chest mein halka sa pressure",
        "seena mein halki jalan", "seena mein thora sa khichao",
        "chest discomfort", "seena mein halki takleef",
    ],
    "mild": [
        "seena mein halka dard", "chest mein halka dard",
        "seena mein chubhan", "seena mein halki jalan",
        "chest mein mild discomfort", "seena mein takleef",
        "seena mein maamuli dard", "seena mein zara si jalan",
    ],
}

# --- duration, graded by severity ---
# A Level-1 crushing pain that has been going on "pichlay kuch dino se"
# is not a Level-1 presentation. Acute tiers get acute durations.
DURATION_BY_SEVERITY = {
    "severe": ["aadhay ghante se", "ek ghante se", "do ghante se",
               "abhi thori der pehle se", "subah se", "raat se",
               "kal raat se", "teen ghante se", "kuch minute pehle se",
               "pandrah minute se", "chaar ghante se"],
    "moderate": ["do ghante se", "teen ghante se", "subah se", "raat se",
                 "kal raat se", "do din se", "kuch dino se",
                 "aadhay ghante se", "kal shaam se", "chaar ghante se",
                 "aaj subah se"],
    "mild_moderate": ["kal se", "do din se", "teen din se", "ek hafte se",
                      "kuch dino se", "pichlay hafte se", "das din se"],
    "mild": ["do din se", "ek hafte se", "kuch dino se",
             "pichlay kuch dino se", "kai hafton se", "kuch mahino se",
             "kaafi arse se", "pichlay mahine se"],
}

# --- cardiac history clauses ---
HISTORY = [
    "known cardiac patient hai", "pehle bhi heart attack ho chuka hai",
    "bypass surgery ho chuki hai", "stent laga hua hai",
    "blood pressure ka mareez hai", "diabetes ka mareez hai",
    "family mein heart disease hai", "hypertension ki history hai",
    "pehle bhi aisa ho chuka hai", "koi purani bimari nahi hai",
    "angioplasty ho chuki hai", "cholesterol high rehta hai",
    "walid ko bhi heart ka masla tha", "smoking ki purani aadat hai",
    "thyroid ka masla hai", "pehle kabhi aisa nahi hua",
]

# --- benign attributions, Level-4 flavour (seeded by the recovered 22) ---
BENIGN_NOTE = [
    "lekin ECG normal hai", "gas jaisi lagti hai",
    "acidity ki wajah se lagta hai", "anxiety attack jaisa lagta hai",
    "muscle kheenchne jaisa lagta hai", "rest se theek ho jata hai",
    "doctor ne kaha koi masla nahi", "pehle bhi hota raha hai aur theek ho jata hai",
    "khana hazam na hone jaisa", "stress ki wajah se lagta hai",
]

SEVERITY_BY_LEVEL = {1: "severe", 2: "moderate",
                     3: "mild_moderate", 4: "mild"}


# ======================================================================
# SENTENCE SKELETONS
#
# Ten genuinely different sentence SHAPES, not one shape with more slots.
# They differ in word order, in which clause leads, in whether the
# patient is the grammatical subject, in punctuation, and in whether the
# sentence is a fragment (how triage notes are actually written) or a
# full sentence. Each returns a complaint string.
# ======================================================================

def _sk_fragment(r, sym, assoc, trig, dur, hist, benign, symnp):
    """Bare note fragment, no verb: 'seena mein dard aur pasina'."""
    return f"{sym} aur {assoc}"


def _sk_symptom_duration(r, sym, assoc, trig, dur, hist, benign, symnp):
    """Symptom then how long: 'seena mein dard do ghante se'."""
    return f"{sym} {dur}"


def _sk_duration_first(r, sym, assoc, trig, dur, hist, benign, symnp):
    """Time leads the sentence: 'Do din se seena mein dard hai.'"""
    return f"{dur.capitalize()} {sym} hai."


def _sk_patient_framed(r, sym, assoc, trig, dur, hist, benign, symnp):
    """Third-person clinical framing: 'Patient ko seena mein dard hai.'"""
    return f"Patient ko {sym} hai {dur} aur {assoc}."


def _sk_trigger_clause(r, sym, assoc, trig, dur, hist, benign, symnp):
    """Trigger leads: 'Seedhiyan chadhte waqt seena mein dard hota hai.'"""
    return f"{trig.capitalize()} {symnp} hota hai."


def _sk_conditional(r, sym, assoc, trig, dur, hist, benign, symnp):
    """Subordinate clause: 'Jab bhi chalta hoon to seena mein dard hota hai.'

    Takes a VERB phrase, not the adverbial trigger - "Jab bhi exercise ke
    baad to ..." is not a sentence. TRIGGER_VERB carries the finite forms.
    """
    return f"Jab bhi {r.choice(TRIGGER_VERB)} to {symnp} hone lagta hai."


def _sk_history_anchored(r, sym, assoc, trig, dur, hist, benign, symnp):
    """History first, complaint second - two clauses, comma separated."""
    return f"{hist.capitalize()}, ab {sym} {dur}."


def _sk_english_clinical(r, sym, assoc, trig, dur, hist, benign, symnp):
    """Terse English note, the way an English-first nurse writes it."""
    eng = {"seena mein dard": "chest pain", "seena bhaari lag raha": "chest heaviness",
           "dhadkan tez": "palpitations", "saans phoolna": "shortness of breath",
           "seena mein jalan": "chest burning"}.get(sym, sym)
    return f"{eng} {trig}, {assoc}"


def _sk_code_switched(r, sym, assoc, trig, dur, hist, benign, symnp):
    """Mid-sentence language switch, very common in real notes."""
    return f"{sym} since {dur.replace(' se', '')}, {assoc}"


def _sk_uncertain(r, sym, assoc, trig, dur, hist, benign, symnp):
    """Hedged / uncertain phrasing: 'Samajh nahi aa raha...'"""
    return f"Samajh nahi aa raha, {sym} hai {trig} aur {assoc}."


def _sk_benign_reassured(r, sym, assoc, trig, dur, hist, benign, symnp):
    """Level-4 shape: complaint plus an explicit benign qualifier."""
    return f"{sym} {trig} {benign}."


def _sk_benign_fragment(r, sym, assoc, trig, dur, hist, benign, symnp):
    """Level-4 fragment, mirrors the recovered 22 almost exactly."""
    return f"{sym} {trig}"


#: Skeletons available to the urgent tiers, and to Level 4. Level 4 gets
#: the two benign shapes plus the neutral ones, so it is not identifiable
#: by sentence shape alone - only by wording and severity.
SKELETONS_URGENT = [
    _sk_fragment, _sk_symptom_duration, _sk_duration_first,
    _sk_patient_framed, _sk_trigger_clause, _sk_conditional,
    _sk_history_anchored, _sk_english_clinical, _sk_code_switched,
    _sk_uncertain,
]
SKELETONS_BENIGN = SKELETONS_URGENT + [
    _sk_benign_reassured, _sk_benign_reassured, _sk_benign_fragment,
]

# ======================================================================
# STRUCTURED FIELDS
#
# ECG and vitals are sampled per level WITH OVERLAP - see the module
# docstring. The weights below are the whole reason a model trained on
# this file cannot reach 99% from ECG_Status alone.
# ======================================================================

# The three infarct patterns (ST elevation, Anterior MI, Inferior MI) appear
# ONLY at Level 1. A STEMI is an emergency regardless of how the complaint is
# worded, and a dataset that put one at Level 2 would be teaching the model
# something clinically false - that is the wrong place to buy label overlap.
# The ambiguity lives instead in the genuinely ambiguous rhythms (Abnormal,
# T wave inversion, Arrhythmia, Sinus tachycardia, Normal), which really do
# span several acuities depending on the patient in front of you.
ECG_BY_LEVEL = {
    1: [("ST elevation", 38), ("Anterior MI", 16), ("Inferior MI", 16),
        ("ST depression", 12), ("Abnormal", 10), ("Arrhythmia", 6),
        ("T wave inversion", 2)],
    2: [("Abnormal", 24), ("T wave inversion", 17), ("Atrial fibrillation", 16),
        ("LBBB", 13), ("ST depression", 12), ("Arrhythmia", 12),
        ("Sinus tachycardia", 6)],
    3: [("Normal", 34), ("Sinus tachycardia", 26), ("Abnormal", 15),
        ("Arrhythmia", 10), ("Bradycardia", 8), ("T wave inversion", 5),
        ("Atrial fibrillation", 2)],
    4: [("Normal", 72), ("Sinus tachycardia", 16), ("Bradycardia", 12)],
}

#: Symptom text that contradicts a rhythm. "Dhadkan tez" (racing heart)
#: alongside a Bradycardia ECG is a self-contradicting record; so is a
#: slow-heart complaint next to Sinus tachycardia.
ECG_TEXT_CONFLICT = {
    "Bradycardia": ("tez", "racing", "irregular", "phoolna"),
}


def _text_conflicts(ecg, text):
    return any(w in text.lower() for w in ECG_TEXT_CONFLICT.get(ecg, ()))

#: (mean, sd) per level. Adjacent levels overlap heavily on purpose;
#: Level 4 is the only tier that is clearly physiologically benign,
#: which matches the 22 recovered Level-4 rows (SpO2 ~98, HR ~100).
VITALS_BY_LEVEL = {
    #      Age        HR          SBP         DBP        Temp        SpO2
    1: [(58, 15), (122, 18), (150, 24), (86, 11), (36.8, 0.3), (89, 4)],
    2: [(55, 16), (114, 17), (145, 23), (83, 11), (36.8, 0.3), (92, 4)],
    3: [(52, 16), (104, 16), (138, 21), (81, 10), (36.8, 0.3), (95, 3)],
    4: [(45, 12), (92, 11), (129, 13), (79, 8), (36.8, 0.2), (97, 1)],
}
VITAL_BOUNDS = [(18, 95), (48, 175), (85, 210), (50, 120), (35.5, 39.5), (78, 100)]
VITAL_INT = [True, True, True, True, False, True]

AVPU_BY_LEVEL = {
    1: [("A", 55), ("V", 32), ("P", 13)],
    2: [("A", 74), ("V", 21), ("P", 5)],
    3: [("A", 93), ("V", 7)],
    4: [("A", 100)],
}
MODE_BY_LEVEL = {
    1: [("Ambulance", 55), ("Wheelchair", 28), ("Walk-in", 17)],
    2: [("Ambulance", 38), ("Wheelchair", 32), ("Walk-in", 30)],
    3: [("Walk-in", 52), ("Wheelchair", 27), ("Ambulance", 21)],
    4: [("Walk-in", 88), ("Wheelchair", 12)],
}
#: Level mix. Level 4 at ~4% matches the proportion the organic dataset
#: carried (407/10000) - see the project roadmap discussion.
LEVEL_MIX = [(1, 30), (2, 40), (3, 26), (4, 4)]


def _weighted(r, pairs):
    return r.choices([p[0] for p in pairs], weights=[p[1] for p in pairs])[0]


# ======================================================================
# TIER BLEEDING
#
# THE BUG THIS FIXES: v2 drew every phrase from the bank matching the
# row's own severity tier, so the banks were effectively disjoint and the
# VOCABULARY gave the label away. Measured on v2: 101 of 277 words
# occurred at exactly one triage level, 76% of rows contained at least
# one of them, and a plain Bag-of-Words model scored 99.85% - it was
# reading the answer off the word list. That is the same class of defect
# as the ECG determinism this generator was written to remove; swapping
# one shortcut for another is not progress.
#
# So a row usually draws from its own tier but sometimes from a
# neighbouring one, with probability falling off by tier distance. Every
# phrase can now appear at several levels, with a frequency gradient
# instead of a hard boundary - which is also how real complaints behave:
# "halka dard" is COMMONER at low acuity, not exclusive to it.
# ======================================================================

TIER_ORDER = ["severe", "moderate", "mild_moderate", "mild"]

#: Weight by |tier distance|. Index 0 is the row's own tier.
TIER_SPREAD = (0.58, 0.18, 0.05, 0.015)

#: Phrases that must NOT bleed into the low-acuity tiers however the dice
#: fall. Syncope or "saans bilkul nahi aa rahi" is not a non-urgent
#: presentation, and a dataset that said so would be teaching the model
#: something dangerous. Matched as substrings against the phrase.
RED_FLAG_FRAGMENTS = (
    "behoshi", "bardasht", "bilkul nahi", "dam ghutne", "phatne",
    "andhera", "girne wala", "chehra sun",
)

#: ...and the mirror case: explicit reassurance must not bleed UP into
#: the emergency tiers, where "rest se theek ho jata hai" would be absurd.
REASSURANCE_FRAGMENTS = (
    "theek ho jata", "koi masla nahi", "baqi sab theek", "zyada takleef nahi",
    "rukawat nahi", "aaram aa jata", "neend theek",
)


def _pick_tiered(r, bank, sev, attempts=12):
    """Draw a phrase from `bank`, usually from tier `sev`, sometimes a neighbour.

    Rejects draws that would put a red-flag phrase at low acuity or a
    reassurance phrase at high acuity, falling back to the row's own tier
    if the dice keep producing blocked combinations.
    """
    i = TIER_ORDER.index(sev)
    tiers = [t for t in TIER_ORDER if t in bank]
    weights = [TIER_SPREAD[min(abs(i - TIER_ORDER.index(t)), len(TIER_SPREAD) - 1)]
               for t in tiers]
    low_acuity = sev in ("mild_moderate", "mild")
    high_acuity = sev in ("severe", "moderate")
    for _ in range(attempts):
        phrase = r.choice(bank[r.choices(tiers, weights=weights)[0]])
        low = phrase.lower()
        if low_acuity and any(f in low for f in RED_FLAG_FRAGMENTS):
            continue
        if high_acuity and any(f in low for f in REASSURANCE_FRAGMENTS):
            continue
        return phrase
    return r.choice(bank[sev])


#: Heart rates an ECG reading physically implies. Without this the
#: sampler happily produced "Bradycardia, HR 126", which is a
#: contradiction in terms and would have taught the model that the two
#: features are unrelated. Rhythms that say nothing about rate are absent
#: and keep the level-based rate.
HR_BY_ECG = {
    "Bradycardia":         (45, 59),
    "Sinus tachycardia":   (101, 145),
    "Atrial fibrillation": (95, 160),
}


def _sample_vitals(r, level, ecg=None):
    out = []
    for (mean, sd), (lo, hi), as_int in zip(VITALS_BY_LEVEL[level],
                                            VITAL_BOUNDS, VITAL_INT):
        v = min(hi, max(lo, r.gauss(mean, sd)))
        out.append(int(round(v)) if as_int else round(v, 1))
    if ecg in HR_BY_ECG:                      # index 1 is Heart_Rate
        lo, hi = HR_BY_ECG[ecg]
        out[1] = r.randint(lo, hi)
    return out


def make_row(r, level, ecg_determinism):
    sev = SEVERITY_BY_LEVEL[level]
    sym = _pick_tiered(r, SYMPTOM, sev)
    symnp = _pick_tiered(r, SYMPTOM_NP, sev)
    assoc = _pick_tiered(r, ASSOCIATED, sev)
    trig = _pick_tiered(r, TRIGGER, sev)
    dur = _pick_tiered(r, DURATION_BY_SEVERITY, sev)
    hist = r.choice(HISTORY)
    benign = r.choice(BENIGN_NOTE)
    # Sentence SHAPE must not give the level away either. Every level can
    # draw every skeleton; the benign shapes are merely likelier at low
    # acuity. In v2 only Level 4 could produce them, which made the tier
    # identifiable from punctuation and clause structure alone.
    benign_weight = {1: 1, 2: 2, 3: 6, 4: 14}[level]
    skeletons = SKELETONS_URGENT + [_sk_benign_reassured, _sk_benign_fragment] * benign_weight
    text = r.choice(skeletons)(r, sym, assoc, trig, dur, hist, benign, symnp)
    text = text[0].upper() + text[1:]

    if ecg_determinism == "strict":
        ecg = {1: "ST elevation", 2: "Abnormal", 3: "Sinus tachycardia",
               4: "Normal"}[level]
    else:
        ecg = _weighted(r, ECG_BY_LEVEL[level])

    age, hr, sbp, dbp, temp, spo2 = _sample_vitals(r, level, ecg)
    return {
        "Age": age,
        "Gender": r.choice(["Male", "Female"]),
        "Mode_of_Arrival": _weighted(r, MODE_BY_LEVEL[level]),
        "Complaint_Text": text,
        "Heart_Rate": hr,
        "Systolic_BP": sbp,
        "Diastolic_BP": dbp,
        "ECG_Status": ecg,
        "Temperature": temp,
        "SpO2": spo2,
        "AVPU": _weighted(r, AVPU_BY_LEVEL[level]),
        "Triage_Level": level,
        "Category": "Cardiac",
    }


def generate(rows, seed, ecg_determinism="realistic", max_attempts_factor=60):
    """Generate `rows` rows with DISTINCT Complaint_Text values.

    Distinctness is enforced by rejection: a duplicate text is discarded
    and resampled. If the skeleton/phrase combinatorics cannot supply
    enough distinct strings the function raises rather than silently
    emitting a shorter or duplicated dataset.
    """
    r = random.Random(seed)
    target = {lvl: round(rows * pct / 100) for lvl, pct in LEVEL_MIX}
    target[1] += rows - sum(target.values())        # absorb rounding

    seen, out = set(), []
    for level, want in target.items():
        got, attempts, limit = 0, 0, want * max_attempts_factor
        while got < want:
            attempts += 1
            if attempts > limit:
                raise RuntimeError(
                    f"Could not produce {want} distinct Level-{level} "
                    f"complaints in {limit} attempts (got {got}). Add "
                    f"phrases or skeletons, or lower --rows.")
            row = make_row(r, level, ecg_determinism)
            if row["Complaint_Text"] in seen:
                continue
            if _text_conflicts(row["ECG_Status"], row["Complaint_Text"]):
                continue        # e.g. "dhadkan tez" recorded under Bradycardia
            seen.add(row["Complaint_Text"])
            out.append(row)
            got += 1
    r.shuffle(out)
    return out


def provenance(rows, seed, ecg_determinism, out_path, generated_on):
    return {
        "synthetic": True,
        "disclosure": (
            "SYNTHETIC DATA - generated by generate_cardiac_dataset.py. "
            "These are not real patient records and must not be described "
            "as such in any publication, evaluation or deployment."),
        "generator": "generate_cardiac_dataset.py",
        "generator_version": GENERATOR_VERSION,
        "generated_on": generated_on,
        "seed": seed,
        "rows": rows,
        "ecg_determinism": ecg_determinism,
        "label_method": (
            "Triage_Level sampled first from LEVEL_MIX; Complaint_Text, "
            "ECG_Status and vitals then sampled conditional on it with "
            "deliberate overlap between adjacent levels, so no single "
            "feature determines the label."),
        "phrase_bank_source": (
            "Fragments derived from the cardiac subset of "
            "triage_mixed_language_dataset_10000_RECOVERED.csv (7,855 rows, "
            "887 distinct complaints, 636-word vocabulary). The 22 genuine "
            "Level-4 complaints in that file seed the Level-4 branch."),
        "output_file": os.path.basename(out_path),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--preview", type=int, default=0,
                   help="print N sample rows and exit WITHOUT writing a file")
    p.add_argument("--ecg-determinism", choices=["realistic", "strict"],
                   default="realistic",
                   help="'strict' reproduces the near-deterministic ECG->level "
                        "mapping of the previous dataset; not recommended")
    p.add_argument("--generated-on", default=None,
                   help="ISO date recorded in the provenance file "
                        "(defaults to today, UTC)")
    args = p.parse_args()

    import pandas as pd

    if args.preview:
        rows = generate(max(args.preview, 400), args.seed, args.ecg_determinism)
        df = pd.DataFrame(rows)
        print(f"PREVIEW - nothing written. Sampling {args.preview} of "
              f"{len(df)} generated rows.\n")
        show = df.sample(args.preview, random_state=args.seed)
        for _, row in show.iterrows():
            print(f"  L{row.Triage_Level} | {row.ECG_Status:<19} | "
                  f"HR {row.Heart_Rate:>3} SpO2 {row.SpO2:>3} | "
                  f"{row.Complaint_Text}")
        print(f"\n  distinct texts in full sample : "
              f"{df.Complaint_Text.nunique()}/{len(df)}")
        print(f"  level mix                     : "
              f"{df.Triage_Level.value_counts().sort_index().to_dict()}")
        vocab = {w for t in df.Complaint_Text
                 for w in ''.join(c if c.isalpha() or c.isspace() else ' '
                                  for c in t.lower()).split()}
        print(f"  vocabulary                    : {len(vocab)} distinct words")
        return

    generated_on = args.generated_on or __import__("datetime").datetime.now(
        __import__("datetime").timezone.utc).strftime("%Y-%m-%d")
    rows = generate(args.rows, args.seed, args.ecg_determinism)
    df = pd.DataFrame(rows)
    out_path = os.path.join(_HERE, args.out) if not os.path.isabs(args.out) else args.out
    df.to_csv(out_path, index=False)

    prov = provenance(args.rows, args.seed, args.ecg_determinism,
                      out_path, generated_on)
    prov_path = out_path + ".provenance.json"
    with open(prov_path, "w", encoding="utf-8") as f:
        json.dump(prov, f, indent=2, ensure_ascii=False)

    print(f"[ok] {len(df)} rows -> {out_path}")
    print(f"[ok] provenance      -> {prov_path}")
    print(f"     distinct texts  : {df.Complaint_Text.nunique()}/{len(df)}")
    print(f"     level mix       : {df.Triage_Level.value_counts().sort_index().to_dict()}")
    print(f"     SYNTHETIC DATA - disclose this in any write-up.")


if __name__ == "__main__":
    main()
