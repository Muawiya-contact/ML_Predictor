# triage_pipeline.py
# ============================================================
# SHARED TRIAGE PIPELINE  (single source of truth)
# ============================================================
# This module holds everything the training script and every
# predictor must agree on:
#
#   * the diacritization dictionary  (variant -> canonical)
#   * the canonical fuzzy-match vocabulary
#   * the domain-guided attention weights
#   * the text normalization pipeline
#   * model / encoder loading + a vectorized predict function
#
# Training (triage_bow_fuzzy_diac.py) and all predictors
# (prediction.py, prediction_interactive.py, predict_batch.py)
# import from here, so the text pipeline can NEVER drift out of
# sync between training and inference.
#
# NOTE: If you edit any dictionary below you MUST retrain
#       (python triage_bow_fuzzy_diac.py) so the saved model,
#       vectorizers and attention weights stay consistent.
# ============================================================

import os
import re
import sys
import numpy as np
import joblib
from rapidfuzz import process, fuzz


# ------------------------------------------------------------
# Console safety (Windows)
# ------------------------------------------------------------

def make_console_safe():
    """Stop Windows' cp1252 console from crashing on diacritized output.

    Canonical forms produced by the diacritization step ("dárd",
    "bukhār", "sēna") cannot be encoded by the default Windows console
    codepage, and printing one raises UnicodeEncodeError. Degrade those
    characters to '?' on screen instead of killing the run - files
    written to disk keep the exact spelling.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass

# ------------------------------------------------------------
# Triage level labels  (model classes are 0..3; dataset is 1..4)
# ------------------------------------------------------------

TRIAGE_LABELS = {
    0: "Level 0 - EMERGENCY   (Immediate attention required)",
    1: "Level 1 - URGENT      (Seen within 15 minutes)",
    2: "Level 2 - STANDARD    (Seen within 60 minutes)",
    3: "Level 3 - NON-URGENT  (Can wait or redirect)",
}

# Short (label, description) pairs for the interactive UI
TRIAGE_LABELS_SHORT = {
    0: ("EMERGENCY",  "Immediate attention required!"),
    1: ("URGENT",     "Seen within 15 minutes"),
    2: ("STANDARD",   "Seen within 60 minutes"),
    3: ("NON-URGENT", "Can wait or be redirected"),
}

# ============================================================
# DIACRITIZATION DICTIONARY
# canonical diacritized form -> list of all known spelling variants
# Inverted at runtime into variant -> canonical for O(1) lookup.
#
# EXPANDED: the original map was almost entirely cardiac. The
# blocks marked "ADDED" widen coverage to the other complaint
# categories that already exist in the dataset (Trauma, Burn,
# Infectious, Neurological, Gyne/Obstetric, Heat, Metabolic,
# plus allergy/skin and psych), so the model can actually learn
# from those complaints instead of treating their words as noise.
# ============================================================

DIACRITIZATION_MAP = {

    # ---------------- PAIN ----------------
    "dárd":     ["dard", "dardh", "durd", "drd", "darad", "darrd",
                 "daard", "daro", "drt"],

    # ---------------- CHEST ----------------
    "sēna":     ["seena", "seenay", "sina", "sena", "seenaa", "sinaa",
                 "syna", "seyna", "chati", "chaati", "chatti"],
    "chest":    ["chst", "chets", "cheast"],

    # ---------------- BREATHING ----------------
    "sāns":     ["saans", "sans", "saaans", "saan", "sanse", "sanss",
                 "breath", "breth", "breeth"],
    "phūlna":   ["phoolna", "phulna", "phulana", "phoolana", "phulraha",
                 "phoolraha", "phulrahi", "phoolrahi", "tangi", "dam", "damghutti"],

    # ---------------- NEURO ----------------
    "bēhōsh":   ["behosh", "behoshy", "behushi", "gash", "gasha",
                 "girpara", "girpari", "unconscious", "unconcious", "unconsius"],
    "chákkar":  ["chakkar", "chakar", "chakr", "chaker", "chakkr",
                 "chakkkar", "dizzy", "gardish"],

    # ---------------- CARDIAC ----------------
    "dháḍkan":  ["dhadkan", "dhadkaan", "dhadkna", "dhadhkan", "dhadkann",
                 "dhakdan", "dhakdhan", "dhak", "dhakdhak",
                 "palpitation", "palpitations", "palpi", "heartbeat", "hrtbt"],

    # ---------------- GI ----------------
    "úlṭī":     ["ulti", "ultee", "ulltee", "ultti", "ulte", "qai", "qay",
                 "qaai", "kaai", "vomit", "vomiting", "vomiit", "nausea", "nausiated"],

    # ---------------- FEVER ----------------
    "bukhār":   ["bukhar", "bukhaar", "bukhr", "bukhur", "buxar", "bkhar",
                 "bkhr", "bukar", "fever", "fevar", "faver", "tap", "garmi"],

    # ---------------- SWEATING ----------------
    "pasīna":   ["pasina", "paseena", "pasiina", "paseenaa", "pasena",
                 "pasnaa", "pasna", "sweat", "sweating", "thandapasina", "coldsweat"],

    # ---------------- WEAKNESS ----------------
    "kamzōrī":  ["kamzori", "kamzoori", "kamzorri", "kamzory", "kamzr",
                 "kmzori", "kmzri", "weak", "weakness", "thakan", "thakaan",
                 "thakawat", "thakaawat", "tired", "tiredness"],

    # ---------------- BODY PARTS ----------------
    "bāzū":     ["bazo", "bazoo", "bazou", "bazu", "bazuu", "baazo", "arm", "arms"],
    "kāndha":   ["kandha", "kanda", "kandaa", "kandhaa", "kndha",
                 "shoulder", "shulder", "shouldr"],
    "gardan":   ["gardann", "gardn", "garrdan", "neck", "nck", "gala", "halaq"],
    "peṭ":      ["pet", "pett", "pait", "paet", "payt", "paait",
                 "stomach", "stomac", "stomak", "abdomen", "belly", "tummy"],
    "sár":      ["sar", "sir", "sarr", "saar", "head", "headache", "sirdard", "sirdrd"],
    "kamár":    ["kamar", "kmar", "kammar", "kamarr"],
    "pīṭh":     ["peeth", "pith", "back", "backpain"],
    "pāir":     ["pair", "paer", "paar", "leg", "legs", "foot", "feet", "taang"],
    "jábra":    ["jabra", "jaabra", "jabraa", "jaw", "jaww"],

    # ---------------- GENERAL SYMPTOMS ----------------
    "sōnn":     ["sonn", "sun", "sunna", "sunnapan", "numbness", "numb", "sunn"],
    "jálan":    ["jalan", "jalann", "jlan", "jaalan", "burning", "burn", "jalana"],
    "sūjan":    ["sujan", "sujaan", "sujn", "sojan", "swelling", "swell",
                 "swolen", "swollen"],
    "khānsī":   ["khansi", "khaansi", "khansy", "khasi", "khaasi",
                 "cough", "coughing", "cofing", "cofgh"],
    "khūn":     ["khoon", "khun", "khonn", "khoonn", "blood", "blod",
                 "bleed", "bleeding"],
    "pēshāb":   ["peshab", "peshaab", "peshabb", "pshab", "urine", "urination", "pee"],

    # ======================================================
    # ============== ADDED: TRAUMA / INJURY ================
    # ======================================================
    "chōṭ":     ["chot", "chott", "injury", "injured", "zakham", "zakhm",
                 "zakhmi", "wound", "wounded", "ghao", "ghaao"],
    "haḍḍī":    ["haddi", "haddii", "hadi", "bone", "fracture", "fractured",
                 "toota", "tuta", "tootgayi", "broken"],
    "girna":    ["girgaya", "girgayi", "fall", "fell", "fallen",
                 "slip", "slipped", "phisalna"],
    "accident": ["acident", "axident", "haadsa", "hadsa", "crash", "takkar"],
    "mōch":     ["moch", "mooch", "sprain", "sprained", "twist", "twisted"],

    # ======================================================
    # ================== ADDED: BURNS ======================
    # ======================================================
    "jálna":    ["jalna", "jalgaya", "jalgayi", "jhulas", "jhulasna",
                 "scald", "scalded", "jhulsa"],

    # ======================================================
    # ============ ADDED: INFECTIOUS / GI BUG ==============
    # ======================================================
    "dast":     ["dasst", "diarrhea", "diarrhoea", "loosemotion",
                 "loose", "motions", "pechish", "pechis"],
    "infection":["infaction", "infction", "sepsis", "septic", "infectn"],

    # ======================================================
    # ============ ADDED: NEURO EMERGENCIES ================
    # ======================================================
    "daura":    ["dora", "doura", "daure", "fit", "fits", "seizure", "siezure",
                 "convulsion", "convulsions", "jhatka", "jhatke", "mirgi", "mirgii"],
    "falij":    ["faalij", "falaj", "paralysis", "paralyzed", "stroke",
                 "strok", "lakwa", "laqwa"],

    # ======================================================
    # ============ ADDED: METABOLIC / ENDOCRINE ============
    # ======================================================
    "shūgar":   ["sugar", "shugar", "shuger", "diabetes", "diabetic",
                 "hypoglycemia", "hyperglycemia", "glucose"],

    # ======================================================
    # ============ ADDED: HEAT-RELATED =====================
    # ======================================================
    "garmīlagna":["heatstroke", "heatstrok", "dehydration",
                  "dehydrated", "sunstroke"],

    # ======================================================
    # ============ ADDED: ALLERGY / SKIN ===================
    # ======================================================
    "kharish":  ["khaarish", "kharsh", "itch", "itching", "rash", "daane",
                 "dane", "allergy", "alergy", "reaction", "chubhan"],

    # ======================================================
    # ============ ADDED: GYNE / OBSTETRIC =================
    # ======================================================
    "haml":     ["hamal", "pregnant", "pregnancy", "pregnent", "delivery",
                 "deliver", "labour", "labor", "zachgi", "zachagi", "miscarriage"],

    # ======================================================
    # ================== ADDED: PSYCH ======================
    # ======================================================
    "ghabrahat":["ghabrahit", "ghabrana", "anxiety", "anxious", "panic",
                 "ghabra", "bechaini", "bechainii"],
}

# Invert at runtime: variant -> canonical  (O(1) lookup)
VARIANT_TO_CANONICAL = {
    variant: canonical
    for canonical, variants in DIACRITIZATION_MAP.items()
    for variant in variants
}

#: Every spelling the diacritization stage can resolve on its own - both the
#: canonical forms and all their listed variants. fuzzy_correct_word() leaves
#: these alone, because an exact dictionary entry beats a fuzzy guess.
_DIACRITIZATION_KNOWN = set(VARIANT_TO_CANONICAL) | set(DIACRITIZATION_MAP)

# ============================================================
# FUZZY MATCHING VOCABULARY
# Canonical (plain) medical terms used to correct unseen
# spelling variants. Only reasonably distinctive words are
# included so that common filler words are not over-corrected.
# ============================================================

CANONICAL_VOCAB = [
    # original cardiac-centric vocabulary
    "dard", "seena", "chest", "saans", "phoolna", "behosh", "chakkar",
    "dhadkan", "bukhar", "pasina", "kamzori", "ulti", "bazo", "kandha",
    "gardan", "pet", "palpitation", "fever", "pain", "unconscious",
    "dizziness", "vomit", "sweat", "heart", "breath", "nausea",
    "sar", "kamar", "peeth", "pair", "jabra", "sonn", "jalan",
    "sujan", "khansi", "khoon", "peshab",
    # ADDED: broader complaint vocabulary
    "chot", "zakhm", "injury", "haddi", "fracture", "accident", "moch",
    "sprain", "jalna", "scald", "dast", "diarrhea", "infection",
    "seizure", "convulsion", "mirgi", "falij", "paralysis", "stroke",
    "sugar", "diabetes", "heatstroke", "dehydration", "allergy", "rash",
    "itching", "pregnancy", "delivery", "miscarriage", "anxiety", "panic",
]

# ============================================================
# DOMAIN-GUIDED LIGHTWEIGHT FEATURE ATTENTION WEIGHTS
# Each BoW feature whose name CONTAINS one of these keys is
# multiplied by the given clinical weight (critical terms are
# boosted; grammatical filler is suppressed).
#
# MATCHING RULES (see build_attention_weights): the longest key
# that matches a feature wins, and keys of 3 characters or fewer
# ("g", "mi", "tha", "pet") only match a WHOLE token. Without
# those two rules the short filler keys at the bottom of this
# dict silently captured most of the vocabulary - "g" alone
# suppressed every feature containing the letter g, including
# "sugar", "girna" and "ghabrahat".
# ============================================================

MEDICAL_WEIGHTS = {
    # ---- PAIN / CHEST ----
    "pain": 2.0, "dard": 2.2, "dárd": 2.2, "seena": 2.8, "sēna": 2.8,
    "chest": 2.5, "tight": 2.2, "pressure": 2.3, "bhaari": 2.0, "boojh": 2.0,

    # ---- ARM / SHOULDER / JAW ----
    "arm": 2.0, "bazo": 2.2, "bāzū": 2.2, "shoulder": 2.0,
    "kandha": 2.0, "kāndha": 2.0, "jaw": 2.2, "gardan": 2.0,

    # ---- BREATHING ----
    "saans": 2.8, "sāns": 2.8, "breath": 2.5, "phoolna": 2.5,
    "phūlna": 2.5, "short": 2.2, "dyspnea": 2.8, "wheezing": 2.3,

    # ---- NEURO ----
    "behosh": 3.0, "bēhōsh": 3.0, "unconscious": 3.0,
    "chakkar": 2.2, "chákkar": 2.2, "dizziness": 2.2, "confusion": 2.2,

    # ---- CARDIAC ----
    "palpitation": 2.2, "dhadkan": 2.3, "dháḍkan": 2.3,
    "heart": 2.5, "mi": 3.0, "arrest": 3.2,

    # ---- GI ----
    "ulti": 2.2, "úlṭī": 2.2, "vomit": 2.2, "nausea": 2.0, "gas": 1.5, "pet": 1.8,

    # ---- SYSTEMIC ----
    "bukhar": 2.0, "bukhār": 2.0, "fever": 2.0,
    "thakan": 1.8, "kamzori": 2.0, "kamzōrī": 2.0,
    "pasina": 2.2, "pasīna": 2.2, "sweat": 2.2, "thanda": 1.8,

    # ======== ADDED: TRAUMA ========
    "chot": 2.2, "chōṭ": 2.2, "zakhm": 2.2, "injury": 2.2, "wound": 2.2,
    "haddi": 2.0, "haḍḍī": 2.0, "fracture": 2.2, "bone": 1.8,
    "accident": 2.6, "haadsa": 2.6, "crash": 2.4,
    "moch": 1.6, "mōch": 1.6, "sprain": 1.6, "girna": 1.8, "fall": 1.8,

    # ======== ADDED: BURNS ========
    "jalna": 2.5, "jálna": 2.5, "jhulas": 2.3, "scald": 2.3, "burn": 2.3,

    # ======== ADDED: INFECTIOUS ========
    "dast": 1.8, "diarrhea": 1.8, "loose": 1.6, "infection": 2.0,
    "sepsis": 3.0, "septic": 3.0,

    # ======== ADDED: NEURO EMERGENCIES ========
    "daura": 2.8, "seizure": 2.8, "convulsion": 2.8, "mirgi": 2.5,
    "falij": 3.0, "paralysis": 3.0, "stroke": 3.0, "lakwa": 3.0,

    # ======== ADDED: METABOLIC ========
    "shugar": 2.0, "shūgar": 2.0, "sugar": 2.0, "diabetes": 2.0,
    "hypoglycemia": 2.8, "glucose": 1.8,

    # ======== ADDED: HEAT ========
    "garmīlagna": 2.2, "heatstroke": 2.6, "dehydration": 2.0,

    # ======== ADDED: ALLERGY / SKIN ========
    "kharish": 1.5, "itch": 1.3, "rash": 1.5, "allergy": 2.0, "reaction": 2.2,

    # ======== ADDED: OBSTETRIC ========
    "haml": 2.3, "pregnan": 2.3, "labour": 2.5, "labor": 2.5,
    "delivery": 2.5, "miscarriage": 2.8,

    # ======== ADDED: PSYCH ========
    "ghabrahat": 1.8, "anxiety": 1.8, "panic": 2.0, "bechaini": 1.6,

    # ---- NOISE REDUCTION ----
    "hai": 0.6, "hain": 0.5, "tha": 0.7, "hey": 0.5, "g": 0.7,
}

# Rule-based replacements applied first (fast, high-confidence)
RULE_REPLACEMENTS = {
    "dardh":             "dard",
    "bazoo":             "bazo",
    "seenay":            "seena",
    "saans phulna":      "saans phoolna",
    "saans phool rahi":  "saans phoolna",
    "dhak dhak":         "palpitation",
    "thanda pasina":     "sweating",
    "pasina":            "sweat",
    "loose motion":      "loosemotion",
    "heat stroke":       "heatstroke",
    "sugar level":       "sugar",
}

# ============================================================
# TEXT NORMALIZATION  (normalization -> fuzzy -> diacritization)
# ============================================================

def fuzzy_correct_word(word, threshold=80):
    """Replace a word with the closest canonical term if score >= threshold.

    A word the diacritization dictionary ALREADY knows exactly is returned
    untouched. Fuzzy matching is a guess for unseen spellings; when the next
    stage has an exact entry for the word, that entry is ground truth and the
    guess can only make it worse.

    THE BUG THIS FIXES: fuzzy matching ran on every token, including ones
    DIACRITIZATION_MAP lists verbatim, and RapidFuzz happily rewrote them to a
    different medical concept because the nearest CANONICAL_VOCAB string is not
    the nearest MEANING:

        sina      (chest)      -> pasina  ->  pasīna      (sweating)
        paseena   (sweating)   -> seena   ->  sēna        (chest)
        crash     (accident)   -> rash    ->  kharish     (itching)
        sunstroke (heat stroke)-> stroke  ->  falij       (paralysis/CVA)

    "paseena" occurs 23 times in triage_mixed_language_dataset.csv, so this
    also mislabelled real training rows, not just live input. STAGE_NOTES
    already documented this guard ("It only fires on spellings the dictionary
    in the next stage does not already know") - it was described but never
    implemented.
    """
    if len(word) < 3:
        return word
    if word in _DIACRITIZATION_KNOWN:
        return word
    match, score, _ = process.extractOne(word, CANONICAL_VOCAB, scorer=fuzz.token_sort_ratio)
    return match if score >= threshold else word


def fuzzy_correct_text(text):
    return ' '.join(fuzzy_correct_word(w) for w in text.split())


def apply_diacritization(text):
    """Map every spelling variant to its canonical diacritized form."""
    return ' '.join(VARIANT_TO_CANONICAL.get(w, w) for w in text.split())


def clean_text(text):
    """Lowercase and replace everything outside a-z / whitespace with a space.

    Runs of spaces are deliberately NOT collapsed here: the rule
    replacements below match on literal single-spaced phrases, and
    collapsing first would make "loose  motion" (two spaces, produced by a
    stripped comma) start matching a rule it does not match today. The
    saved Bag-of-Words vectorizers in triage_model/ were fitted under the
    current behaviour, so it is preserved exactly.
    """
    return re.sub(r"[^a-z\s]", " ", str(text).lower())


def apply_rule_replacements(text):
    """High-confidence multi-word fixes applied before fuzzy matching.

    This is a REAL pipeline stage - it is what turns "saans phulna" into
    "saans phoolna", "thanda pasina" into "sweating" and "sugar level"
    into "sugar". Anything that shows the pipeline to a human must show
    this stage too, or words appear to vanish later for no visible reason.
    """
    for k, v in RULE_REPLACEMENTS.items():
        text = text.replace(k, v)
    return text


#: The normalization pipeline as an ordered list of (key, title, function).
#: normalize_roman_urdu() and every explain-the-pipeline view are both built
#: from this list, so a stage can never be shown that is not actually run,
#: and a stage can never be run that is not shown.
NORMALIZATION_STAGES = [
    ("clean", "Lowercase + punctuation stripped", clean_text),
    ("rules", "Rule-based phrase replacements", apply_rule_replacements),
    ("fuzzy", "Fuzzy spelling correction", fuzzy_correct_text),
    ("diacritize", "Diacritization", apply_diacritization),
]

STAGE_NOTES = {
    "clean": "everything except a-z becomes a space",
    "rules": ("fixed multi-word phrases from RULE_REPLACEMENTS "
              "(saans phulna -> saans phoolna, sugar level -> sugar)"),
    "fuzzy": ("RapidFuzz maps an unknown spelling onto the closest word in "
              "CANONICAL_VOCAB at >=80% similarity. It only fires on spellings "
              "the dictionary in the next stage does not already know, so on "
              "many complaints it correctly changes nothing."),
    "diacritize": ("every spelling variant listed in DIACRITIZATION_MAP "
                   "collapses to one canonical form"),
    "stopwords": ("Contribution 1 - the list was learned from the data. Only "
                  "tokens that are BOTH common AND statistically unrelated to "
                  "the triage level are dropped; common words that do track "
                  "triage (\"hai\", \"aur\", \"mein\") are kept on purpose."),
}


def normalize_roman_urdu(text):
    """Full text pipeline: clean -> rule replace -> fuzzy -> diacritize."""
    for _, _, fn in NORMALIZATION_STAGES:
        text = fn(text)
    return text


def normalize_stages(text):
    """Run the pipeline and return what every stage produced.

    Returns a list of dicts: key, title, text, note, changed. The last
    entry's "text" is exactly normalize_roman_urdu(text) - the same code
    runs both, so an explainer built on this cannot drift from the model.
    """
    stages = [{"key": "raw", "title": "Raw input", "text": str(text),
               "note": "exactly what the triage nurse typed", "changed": False}]
    current = text
    previous = " ".join(str(text).split())
    for key, title, fn in NORMALIZATION_STAGES:
        current = fn(current)
        shown = " ".join(current.split())
        stages.append({"key": key, "title": title, "text": shown,
                       "note": STAGE_NOTES.get(key, ""),
                       "changed": shown != previous})
        previous = shown
    return stages


def preprocess_stages(text, stopword_set=None):
    """normalize_stages() plus the learned stop-word removal stage.

    The final entry's "text" is exactly preprocess_for_embedding(text).
    """
    from stopwords import load_stopwords, remove_stopwords

    if stopword_set is None:
        stopword_set = load_stopwords()

    stages = normalize_stages(text)
    normalized = stages[-1]["text"]
    final = remove_stopwords(normalize_roman_urdu(text), stopword_set)
    shown = " ".join(final.split())
    stages.append({"key": "stopwords", "title": "Learned stop words removed",
                   "text": shown, "note": STAGE_NOTES["stopwords"],
                   "changed": shown != normalized,
                   # dict.fromkeys de-duplicates while keeping the order they
                   # appeared in, so a complaint saying "se ... se" lists the
                   # token once rather than twice in the explanation table.
                   "dropped": list(dict.fromkeys(
                       w for w in normalized.split() if w in stopword_set))})
    return stages


# Backwards-compatible alias used by the interactive script
normalize = normalize_roman_urdu


# ============================================================
# PREPROCESSING FOR THE EMBEDDING PATH  (ARCHITECTURE.md, Tasks 1-2)
#
#   raw text -> lowercase/clean -> fuzzy normalize -> remove
#   learned stop words -> embed
#
# The fuzzy spelling step is deliberately KEPT (Task 2): it is what
# collapses bukhar / bukhaar / bukharr into one form, which a sentence
# embedding model trained mostly on English and native-script Urdu will
# not do on its own for Roman Urdu.
#
# WHY THIS IS A SEPARATE FUNCTION, not a flag flipped on
# normalize_roman_urdu():
#   The saved Bag-of-Words model in triage_model/ was fitted on text
#   WITHOUT stop-word removal. Changing normalize_roman_urdu() would
#   feed the existing vectorizers text they were never fitted on and
#   silently degrade predict_batch.py / prediction.py /
#   prediction_interactive.py. The dictionary+BoW path therefore keeps
#   its exact current behaviour, and the new embedding path gets its
#   own preprocessing entry point.
# ============================================================

def preprocess_for_embedding(text, stopword_set=None):
    """Full preprocessing for the embedding path.

    Args:
        text: raw complaint string.
        stopword_set: learned stop words. Pass None to load the saved
            learned_stopwords.json; pass an explicit set (including an
            empty one) to control it directly.

    Returns:
        Normalized, fuzzy-corrected, stop-word-stripped text.
    """
    # Imported lazily: stopwords.py imports this module for its clinical
    # safety guard, so a module-level import here would be circular.
    from stopwords import load_stopwords, remove_stopwords

    if stopword_set is None:
        stopword_set = load_stopwords()

    return remove_stopwords(normalize_roman_urdu(text), stopword_set)


def preprocess_corpus_for_embedding(texts, stopword_set=None):
    """Vectorized preprocess_for_embedding over an iterable of complaints.

    Loads the stop-word list once instead of per row.
    """
    from stopwords import load_stopwords

    if stopword_set is None:
        stopword_set = load_stopwords()

    return [preprocess_for_embedding(t, stopword_set) for t in texts]


# ============================================================
# ATTENTION
# ============================================================

#: Minimum key length for substring matching. Anything shorter is a
#: filler word and must match a whole token instead.
_MIN_SUBSTRING_KEY_LEN = 4

#: Longest key first, so the most specific medical term wins over a short
#: filler key that happens to be a substring of it.
_ORDERED_WEIGHTS = sorted(MEDICAL_WEIGHTS.items(), key=lambda kv: -len(kv[0]))


def build_attention_weights(feature_names):
    """Return the per-feature weight vector for a list of BoW feature names.

    A feature takes the weight of the LONGEST matching key, and the search
    stops there. Short keys (< 4 characters) are the grammatical filler at
    the bottom of MEDICAL_WEIGHTS, so they only match a whole token.

    The previous version scanned MEDICAL_WEIGHTS in insertion order with no
    break, so the LAST matching key won - and the last key in the dict is
    "g": 0.7. Every feature containing the letter g was therefore suppressed
    to 0.7 regardless of its clinical weight ("sugar" 2.0 -> 0.7,
    "garmīlagna" 2.2 -> 0.7, "girna" 1.8 -> 0.7), and unlisted features such
    as "bleeding" or "emergency" were suppressed too instead of staying at
    the neutral 1.0. "tha": 0.7 and "mi": 3.0 did the same to "thanda" and
    "vomit".
    """
    weights = np.ones(len(feature_names))
    for i, feat in enumerate(feature_names):
        tokens = feat.split()
        for k, w in _ORDERED_WEIGHTS:
            if len(k) >= _MIN_SUBSTRING_KEY_LEN:
                matched = k in feat
            else:
                matched = k in tokens
            if matched:
                weights[i] = w
                break
    return weights


def apply_attention(text_matrix, feature_names):
    return text_matrix * build_attention_weights(feature_names)


# ============================================================
# MODEL LOADING + SAFE ENCODING
# ============================================================

NUMERICAL_FEATURES = ['Age', 'Heart_Rate', 'Systolic_BP',
                      'Diastolic_BP', 'Temperature', 'SpO2']

# Columns a prediction input row must provide
REQUIRED_INPUT_COLUMNS = [
    'Complaint_Text', 'Age', 'Gender', 'Mode_of_Arrival',
    'Heart_Rate', 'Systolic_BP', 'Diastolic_BP',
    'Temperature', 'SpO2', 'AVPU', 'ECG_Status',
]


# ============================================================
# MODEL BUNDLE MANIFEST
#
# A saved model directory describes ITSELF. Before this existed, every
# predictor assumed "model.pkl was trained on structured + attention-weighted
# Bag-of-Words", which is why an embedding-based model could not be deployed
# at all: nothing downstream could build the features it expects. The
# manifest names the text representation, so load_artifacts() can assemble
# the right feature blocks in the right order.
#
# A directory with NO manifest is the historical dictionary+BoW bundle
# (triage_model/), and is loaded exactly as before.
# ============================================================

MANIFEST_FILE = 'model_manifest.json'

LEGACY_MANIFEST = {
    'text_representation': 'dictionary_bow',
    'method': 'A) Dictionary + BoW',
    'embedding_model': None,
    'embedding_dim': None,
    'text_pipeline': 'clean -> rule replace -> fuzzy -> diacritize -> BoW + attention',
}

#: Which feature blocks each representation stacks, in training order.
#: MUST match the np.hstack order in train_embedding_pipeline.py.
REPRESENTATION_BLOCKS = {
    'dictionary_bow':          ('bow',),
    'embeddings_raw':          ('embedding',),
    'embeddings_preprocessed': ('embedding',),
    'hybrid':                  ('bow', 'embedding'),
}


def read_manifest(model_dir):
    """Describe a saved model directory. Falls back to the legacy layout."""
    path = os.path.join(model_dir, MANIFEST_FILE)
    if not os.path.exists(path):
        return dict(LEGACY_MANIFEST)
    import json
    with open(path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    for k, v in LEGACY_MANIFEST.items():
        manifest.setdefault(k, v)
    return manifest


def describe_model(model_dir='triage_model'):
    """Short human-readable description of what a model directory contains.

    Used by the GUI and the CLI predictors so the operator can always see
    WHICH method is actually making the prediction in front of them.
    """
    manifest = read_manifest(model_dir)
    rep = manifest['text_representation']
    blocks = REPRESENTATION_BLOCKS.get(rep, ())
    if 'embedding' in blocks and 'bow' in blocks:
        basis = 'dictionary BoW + sentence-transformer embeddings'
    elif 'embedding' in blocks:
        basis = 'sentence-transformer embeddings'
    else:
        basis = 'dictionary + Bag-of-Words'
    return {
        'model_dir': model_dir,
        'method': manifest.get('method') or rep,
        'text_representation': rep,
        'basis': basis,
        'uses_embeddings': 'embedding' in blocks,
        'embedding_model': manifest.get('embedding_model'),
        'embedding_dim': manifest.get('embedding_dim'),
    }


#: The two model bundles this project ships. The embedding bundle is the
#: deployed one; the dictionary bundle is the offline-safe fallback for a
#: machine without sentence-transformers installed.
EMBEDDING_MODEL_DIR = 'triage_model_embedding'
DICTIONARY_MODEL_DIR = 'triage_model'


def resolve_model_dir(model_dir=None, allow_fallback=True):
    """Decide which saved model actually runs, and say why.

    Returns (model_dir, note). An explicit model_dir is always honoured.
    Otherwise the embedding bundle wins, unless it needs
    sentence-transformers and that is not installed - in which case the
    dictionary bundle is used and the note says so, rather than the caller
    crashing on an import deep inside a prediction.
    """
    if model_dir:
        return model_dir, ''

    candidate = EMBEDDING_MODEL_DIR
    if not os.path.exists(os.path.join(candidate, 'model.pkl')):
        return DICTIONARY_MODEL_DIR, (
            f"'{candidate}/' not found - using the dictionary model. "
            "Run: python train_embedding_pipeline.py")

    info = describe_model(candidate)
    if info['uses_embeddings'] and allow_fallback:
        try:
            import sentence_transformers          # noqa: F401
        except ImportError:
            return DICTIONARY_MODEL_DIR, (
                f"'{candidate}/' needs sentence-transformers "
                f"({info['embedding_model']}), which is not installed - "
                "falling back to the dictionary model in "
                f"'{DICTIONARY_MODEL_DIR}/'. Install it with: "
                "pip install -r requirements-embedding.txt")
    return candidate, ''


def load_artifacts(model_dir='triage_model'):
    """Load the trained model, vectorizers, scaler, encoders and manifest."""
    def p(name):
        return os.path.join(model_dir, name)

    manifest = read_manifest(model_dir)
    rep = manifest['text_representation']
    if rep not in REPRESENTATION_BLOCKS:
        raise ValueError(
            f"{model_dir}/{MANIFEST_FILE} declares unknown text_representation "
            f"'{rep}'. Known: {sorted(REPRESENTATION_BLOCKS)}")
    blocks = REPRESENTATION_BLOCKS[rep]

    artifacts = {
        'model':     joblib.load(p('model.pkl')),
        'scaler':    joblib.load(p('scaler.pkl')),
        'le_gender': joblib.load(p('gender_enc.pkl')),
        'le_mode':   joblib.load(p('mode_enc.pkl')),
        'le_avpu':   joblib.load(p('avpu_enc.pkl')),
        'le_ecg':    joblib.load(p('ecg_enc.pkl')),
        'manifest':  manifest,
        'model_dir': model_dir,
        'text_representation': rep,
        'blocks': blocks,
    }

    if 'bow' in blocks:
        artifacts['word_bow'] = joblib.load(p('word_bow.pkl'))
        artifacts['char_bow'] = joblib.load(p('char_bow.pkl'))
        artifacts['feature_names'] = (
            list(artifacts['word_bow'].get_feature_names_out()) +
            list(artifacts['char_bow'].get_feature_names_out())
        )
        artifacts['attention'] = build_attention_weights(artifacts['feature_names'])

    # Per-block rescaling, saved only by the hybrid path (see
    # train_embedding_pipeline.py). Without it the attention-weighted BoW
    # block (values around 8) drowns out the L2-normalized embedding block
    # (values around 0.05) and the classifier ignores the embeddings.
    for block in blocks:
        scaler_path = p(f'{block}_block_scaler.pkl')
        if os.path.exists(scaler_path):
            artifacts[f'{block}_block_scaler'] = joblib.load(scaler_path)

    artifacts['encoder'] = None          # loaded lazily on first use
    return artifacts


def get_text_encoder(art):
    """Return the sentence-transformer for this bundle, loading it once."""
    if art.get('encoder') is not None:
        return art['encoder']
    name = art['manifest'].get('embedding_model')
    if not name:
        raise RuntimeError(
            f"{art['model_dir']} is an embedding model but its manifest does "
            "not name an embedding_model. Re-run train_embedding_pipeline.py.")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise RuntimeError(
            f"The deployed model in '{art['model_dir']}' uses sentence "
            f"embeddings ({name}), so 'sentence-transformers' is required.\n"
            "Install it once (needs internet), then re-run:\n"
            "    pip install -r requirements-embedding.txt"
        ) from e
    art['encoder'] = SentenceTransformer(name, device='cpu')
    return art['encoder']


def build_text_features(art, raw_texts, batch_size=32):
    """Turn raw complaint strings into the text feature block this model wants.

    The block order here MUST match the np.hstack order used at training
    time (bow first, then embeddings) - see REPRESENTATION_BLOCKS.
    """
    raw_texts = ['unknown' if t is None else str(t) for t in raw_texts]
    rep = art['text_representation']
    parts = []

    for block in art['blocks']:
        if block == 'bow':
            cleaned = [normalize_roman_urdu(t) for t in raw_texts]
            mat = np.hstack([
                art['word_bow'].transform(cleaned).toarray(),
                art['char_bow'].transform(cleaned).toarray(),
            ]) * art['attention']
        else:
            # 'embeddings_raw' deliberately skips preprocessing; every other
            # embedding representation gets clean -> fuzzy -> stop-word removal,
            # exactly as train_embedding_pipeline.py did.
            texts = (list(raw_texts) if rep == 'embeddings_raw'
                     else preprocess_corpus_for_embedding(raw_texts))
            mat = get_text_encoder(art).encode(
                texts, batch_size=batch_size, show_progress_bar=False,
                convert_to_numpy=True, normalize_embeddings=True)
        block_scaler = art.get(f'{block}_block_scaler')
        if block_scaler is not None:
            mat = block_scaler.transform(mat)
        parts.append(mat)

    return np.hstack(parts)


# ============================================================
# READING PATIENT FILES
# ============================================================

#: Tried in order when a CSV does not decode as UTF-8. cp1252 comes before
#: latin-1 because latin-1 decodes ANY byte sequence without error, so
#: putting it earlier would silently turn Windows curly quotes and other
#: cp1252 bytes into mojibake instead of letting cp1252 read them properly.
CSV_ENCODINGS = ('utf-8', 'utf-8-sig', 'cp1252', 'latin-1')


def read_table(path):
    """Read a CSV or Excel file of patients into a DataFrame.

    Excel files carry their own encoding, but a CSV exported from Excel on a
    Windows machine is usually cp1252/latin-1, not UTF-8, and pandas raised
    UnicodeDecodeError on the first non-ASCII byte. A triage file must not be
    rejected because someone typed a degree sign, so the encoding is detected
    (chardet when available) and then a fallback chain is tried.
    """
    import pandas as pd

    ext = os.path.splitext(path)[1].lower()
    if ext in ('.xlsx', '.xls'):
        return pd.read_excel(path)
    if ext not in ('.csv', '.txt'):
        raise ValueError(f"Unsupported file type '{ext}'. Use .xlsx or .csv")

    candidates = []
    try:                                    # optional; never required
        import chardet
        with open(path, 'rb') as f:
            guess = chardet.detect(f.read(200_000))
        if guess.get('encoding') and guess.get('confidence', 0) >= 0.6:
            candidates.append(guess['encoding'])
    except Exception:
        pass
    candidates += [e for e in CSV_ENCODINGS if e not in candidates]

    last_error = None
    for encoding in candidates:
        try:
            return pd.read_csv(path, encoding=encoding)
        except (UnicodeDecodeError, LookupError) as e:
            last_error = e
    # latin-1 decodes any byte, so reaching here means the file is not the
    # text file it claims to be. Say that instead of re-raising a decode error.
    raise ValueError(
        f"Could not read '{path}' as text with any of {candidates}. "
        f"Is it really a CSV? Underlying error: {last_error}")


def safe_encode(encoder, value, warnings_list=None, field=''):
    """
    Encode a categorical value. If the value was not seen during
    training, fall back to the first known class and record a warning,
    instead of crashing (important for messy batch files).
    """
    value = str(value).strip()
    classes = list(encoder.classes_)
    if value in classes:
        return int(encoder.transform([value])[0])
    # case-insensitive rescue
    lower_map = {c.lower(): c for c in classes}
    if value.lower() in lower_map:
        return int(encoder.transform([lower_map[value.lower()]])[0])
    if warnings_list is not None:
        warnings_list.append(
            f"{field}='{value}' not seen in training -> defaulted to '{classes[0]}'")
    return int(encoder.transform([classes[0]])[0])


def _safe_float(value, default):
    """Coerce a value to float. Returns (value, was_substituted)."""
    try:
        f = float(value)
        if np.isnan(f):
            return default, True
        return f, False
    except (TypeError, ValueError):
        return default, True


# ============================================================
# PREDICTION  (single + vectorized batch)
# ============================================================

def predict_one(art, complaint, age, heart_rate, systolic_bp, diastolic_bp,
                temperature, spo2, gender, mode_of_arrival, avpu, ecg_status):
    """Predict triage for a single patient. Returns (level, confidence, proba)."""
    text_feat = build_text_features(art, [complaint])

    import pandas as pd
    numerical = art['scaler'].transform(pd.DataFrame(
        [[age, heart_rate, systolic_bp, diastolic_bp, temperature, spo2]],
        columns=NUMERICAL_FEATURES))
    categorical = np.array([[
        safe_encode(art['le_gender'], gender),
        safe_encode(art['le_mode'],   mode_of_arrival),
        safe_encode(art['le_avpu'],   avpu),
        safe_encode(art['le_ecg'],    ecg_status),
    ]])

    X = np.hstack([numerical, categorical, text_feat])
    proba = art['model'].predict_proba(X)[0]
    level = int(np.argmax(proba))
    return level, float(proba[level]), proba


def predict_dataframe(art, df):
    """
    Vectorized batch prediction over a pandas DataFrame containing the
    columns in REQUIRED_INPUT_COLUMNS. Missing numeric values are filled
    with the training means; unseen categoricals fall back safely.

    Returns (results_df, per_row_warnings) where results_df adds:
        Predicted_Level_0to3, Predicted_Triage_Level (1-4),
        Predicted_Label, Confidence, P_L0..P_L3, Notes
    """
    import pandas as pd

    df = df.copy()

    # --- ensure all required columns exist ---
    for col in REQUIRED_INPUT_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    # --- numeric fill (use training means from the scaler) ---
    means = {f: m for f, m in zip(NUMERICAL_FEATURES, art['scaler'].mean_)}
    row_notes = ['' for _ in range(len(df))]

    # Note EVERY substituted vital sign, not just blank/NaN ones. A typo such
    # as Heart_Rate="l10" also falls back to the training mean, and previously
    # that row came out with an empty Notes column - an operator would see a
    # triage level computed from a vital sign the patient never had, with no
    # indication anything was replaced.
    num_matrix = np.zeros((len(df), len(NUMERICAL_FEATURES)))
    for j, f in enumerate(NUMERICAL_FEATURES):
        for i, raw in enumerate(df[f].tolist()):
            val, substituted = _safe_float(raw, means[f])
            if substituted:
                blank = raw is None or (isinstance(raw, float) and np.isnan(raw))
                reason = "missing" if blank else f"unreadable ('{raw}')"
                row_notes[i] += f"{f} {reason}->mean; "
            num_matrix[i, j] = val
    num_matrix = art['scaler'].transform(
        pd.DataFrame(num_matrix, columns=NUMERICAL_FEATURES))

    # --- text features (vectorize / embed the whole column at once) ---
    text_feat = build_text_features(
        art, df['Complaint_Text'].fillna('unknown').astype(str).tolist())

    # --- categorical encoding (safe) ---
    cat_matrix = np.zeros((len(df), 4))
    enc_specs = [
        ('le_gender', 'Gender',          'Gender'),
        ('le_mode',   'Mode_of_Arrival', 'Mode_of_Arrival'),
        ('le_avpu',   'AVPU',            'AVPU'),
        ('le_ecg',    'ECG_Status',      'ECG_Status'),
    ]
    for j, (enc_key, col, field) in enumerate(enc_specs):
        for i, raw in enumerate(df[col].tolist()):
            warns = []
            cat_matrix[i, j] = safe_encode(art[enc_key], raw, warns, field)
            if warns:
                row_notes[i] += warns[0] + "; "

    # --- combine + predict ---
    X = np.hstack([num_matrix, cat_matrix, text_feat])
    proba = art['model'].predict_proba(X)
    levels = np.argmax(proba, axis=1)

    out = df.copy()
    out['Predicted_Level_0to3']   = levels.astype(int)
    out['Predicted_Triage_Level'] = (levels + 1).astype(int)
    out['Predicted_Label']        = [TRIAGE_LABELS[int(l)].split('(')[0].strip()
                                     for l in levels]
    out['Confidence']             = [f"{proba[i, levels[i]]*100:.1f}%"
                                     for i in range(len(df))]
    for k in range(4):
        out[f'P_L{k}'] = [f"{proba[i, k]*100:.1f}%" for i in range(len(df))]
    out['Notes'] = [n.strip().rstrip(';') for n in row_notes]

    return out, row_notes
