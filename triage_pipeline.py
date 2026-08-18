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
# Where the project's own files live
#
# Every asset this project ships - the two model bundles, the learned
# stop-word list, the dataset, the results CSVs - used to be named by a
# bare relative path ('triage_model', 'learned_stopwords.json', ...).
# A bare relative path is resolved against the CURRENT WORKING
# DIRECTORY, not against the code, so the whole project only worked
# when it happened to be launched from its own folder. Run any script
# from anywhere else - an IDE with a different workspace root, a
# desktop shortcut, a scheduled job, `cd ..` - and two things happened:
#
#   * load_artifacts() died with
#     FileNotFoundError: 'triage_model/model.pkl', and
#   * load_stopwords() silently returned an EMPTY set, because it is
#     written to degrade gracefully on a fresh clone. That one is the
#     dangerous case: the model was TRAINED with stop-word removal, so
#     inference then ran a different text pipeline than training, with
#     no error anywhere - just quietly different triage levels.
#
# Anchoring to __file__ makes the location of the code, not the
# location of the terminal, decide where the project's files are.
# ------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def project_path(*parts):
    """Absolute path to a file that ships with this project."""
    return os.path.join(PROJECT_ROOT, *parts)


def resolve_project_file(path):
    """Resolve a project asset without breaking caller-supplied paths.

    A path that is absolute, or that exists relative to the current
    directory, is honoured exactly as given - so an operator can still
    keep a local override next to their own working files, and a file
    picked in a file dialog is never redirected. Only when the path
    does not resolve against the cwd do we fall back to the copy that
    ships with the project.
    """
    if not path or os.path.isabs(path) or os.path.exists(path):
        return path
    shipped = project_path(path)
    return shipped if os.path.exists(shipped) else path


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
                 "daard", "daro", "drt",
                 # English synonyms: a nurse typing "pain" must land on the
                 # same canonical token as one typing "drd" - the embedding
                 # only weakly clusters cross-language synonyms (55.6% pass
                 # rate for chest_pain in embedding_evaluation_results.csv),
                 # so the collapse happens here, not at the embedding stage.
                 "pain", "pains", "ache", "aches", "aching",
                 "hurt", "hurts", "hurting", "painful"],

    # ---------------- CHEST ----------------
    # "chest" used to be its own canonical, so English "chest pain" and Roman
    # Urdu "seena mein dard" never met - 3279 and 4215 rows of the same
    # complaint sitting in two different tokens. Merged here rather than at
    # inference time, because both spellings are frequent in training.
    "sēna":     ["seena", "seenay", "seene", "sina", "sena", "seenaa", "sinaa",
                 "syna", "seyna", "chati", "chaati", "chatti",
                 "chest", "chst", "chets", "cheast", "thorax", "thoracic"],

    # ---------------- BREATHING ----------------
    "sāns":     ["saans", "sans", "saaans", "saan", "sanse", "sanss",
                 "breath", "breth", "breeth",
                 "breathing", "breathless", "breathlessness",
                 "dyspnea", "dyspnoea", "sob"],
    "phūlna":   ["phoolna", "phulna", "phulana", "phoolana", "phulraha",
                 "phoolraha", "phulrahi", "phoolrahi", "tangi", "dam", "damghutti",
                 "phool", "phoolta", "phulta", "phoolti", "phulti"],

    # ---------------- NEURO ----------------
    "bēhōsh":   ["behosh", "behoshy", "behushi", "gash", "gasha",
                 "girpara", "girpari", "unconscious", "unconcious", "unconsius",
                 # "collapse"/"blackout" deliberately NOT here: they can
                 # describe a mechanical fall rather than true loss of
                 # consciousness, and this token feeds a 3.0 attention weight.
                 "faint", "fainted", "fainting", "syncope",
                 # "behoshi" is 506 rows of cardiac_multilingual_10000.csv and
                 # was the single largest normalization hole in the pipeline:
                 # the map knew behosh/behoshy/behushi but not this spelling.
                 "behoshi", "behoshee", "unresponsive", "obtunded"],
    "chákkar":  ["chakkar", "chakar", "chakr", "chaker", "chakkr",
                 "chakkkar", "dizzy", "gardish",
                 "dizziness", "vertigo", "lightheaded", "giddy", "giddiness"],

    # ---------------- CARDIAC ----------------
    # "dil" is the canonical for the organ. English "heart" was never mapped
    # onto it, so "heart may pain hay" and "dil mein dard hai" reached the
    # encoder as different tokens. Both spellings are frequent in
    # cardiac_multilingual_10000.csv (heart 812 rows, dil 1005), which is why
    # this mapping ships WITH a retrain - applying it at inference only would
    # have moved live input away from what the model was fitted on.
    "dil":      ["heart", "hart", "herat", "dill", "dl", "cardiac", "cardio"],
    "dháḍkan":  ["dhadkan", "dhadkaan", "dhadkna", "dhadhkan", "dhadkann",
                 "dhakdan", "dhakdhan", "dhak", "dhakdhak",
                 "palpitation", "palpitations", "palpi", "heartbeat", "hrtbt",
                 "tachycardia", "bradycardia", "fluttering", "flutter",
                 "pounding", "thumping",
                 "beat", "beats", "dhadak", "dhadakna", "dhadakta", "dhadakti",
                 "dharakna", "dharakta", "dharakti", "dhadkanein",
                 # "racing" occurs 0 times in cardiac_multilingual_10000_v3.csv,
                 # so mapping it cannot desync inference from training - the
                 # same safety argument as the "hay" -> "hai" fix. It closes
                 # the one same-meaning pair that failed the 0.5 threshold
                 # ("dil ki dhadkan tez" vs "heart racing fast").
                 "racing"],

    # ---------------- GI ----------------
    "úlṭī":     ["ulti", "ultee", "ulltee", "ultti", "ulte", "qai", "qay",
                 "qaai", "kaai", "vomit", "vomiting", "vomiit", "nausea", "nausiated",
                 "vomited", "vomits", "puke", "puking",
                 "nauseous", "nauseated", "retching"],

    # ---------------- FEVER ----------------
    "bukhār":   ["bukhar", "bukhaar", "bukhr", "bukhur", "buxar", "bkhar",
                 "bkhr", "bukar", "fever", "fevar", "faver", "tap", "garmi",
                 "feverish", "febrile", "pyrexia", "temperature"],

    # ---------------- SWEATING ----------------
    "pasīna":   ["pasina", "paseena", "pasiina", "paseenaa", "pasena",
                 "pasnaa", "pasna", "sweat", "sweating", "thandapasina", "coldsweat",
                 "sweats", "sweaty", "perspiration", "diaphoresis"],

    # ---------------- WEAKNESS ----------------
    "kamzōrī":  ["kamzori", "kamzoori", "kamzorri", "kamzory", "kamzr",
                 "kmzori", "kmzri", "weak", "weakness", "thakan", "thakaan",
                 "thakawat", "thakaawat", "tired", "tiredness",
                 "fatigue", "fatigued", "lethargy", "lethargic",
                 "exhausted", "exhaustion"],

    # ---------------- BODY PARTS ----------------
    "bāzū":     ["bazo", "bazoo", "bazou", "bazu", "bazuu", "baazo", "arm", "arms"],
    "kāndha":   ["kandha", "kanda", "kandaa", "kandhaa", "kndha",
                 "shoulder", "shulder", "shouldr",
                 "kandhay", "kandhe", "kandhon", "shoulders"],
    "hāth":     ["haath", "hath", "haathon", "hathon",
                 "hand", "hands", "wrist", "forearm"],
    "gardan":   ["gardann", "gardn", "garrdan", "neck", "nck", "gala", "halaq",
                 "throat"],
    "peṭ":      ["pet", "pett", "pait", "paet", "payt", "paait",
                 "stomach", "stomac", "stomak", "abdomen", "belly", "tummy",
                 "abdominal"],
    # "migraine" is a diagnosis, not a synonym for "head"; collapsing it here
    # would throw away the signal that distinguishes it from generic headache.
    "sár":      ["sar", "sir", "sarr", "saar", "head", "headache", "sirdard", "sirdrd"],
    "kamár":    ["kamar", "kmar", "kammar", "kamarr"],
    "pīṭh":     ["peeth", "pith", "back", "backpain"],
    "pāir":     ["pair", "paer", "paar", "leg", "legs", "foot", "feet", "taang"],
    "jábra":    ["jabra", "jaabra", "jabraa", "jaw", "jaww"],

    # ---------------- GENERAL SYMPTOMS ----------------
    "sōnn":     ["sonn", "sun", "sunna", "sunnapan", "numbness", "numb", "sunn"],
    "jálan":    ["jalan", "jalann", "jlan", "jaalan", "burning", "burn", "jalana",
                 "heartburn"],
    "sūjan":    ["sujan", "sujaan", "sujn", "sojan", "swelling", "swell",
                 "swolen", "swollen", "edema", "oedema"],
    "khānsī":   ["khansi", "khaansi", "khansy", "khasi", "khaasi",
                 "cough", "coughing", "cofing", "cofgh"],
    "khūn":     ["khoon", "khun", "khonn", "khoonn", "blood", "blod",
                 "bleed", "bleeding", "hemorrhage", "haemorrhage"],
    "pēshāb":   ["peshab", "peshaab", "peshabb", "pshab", "urine", "urination", "pee",
                 "urinary"],

    # ======================================================
    # ============== ADDED: TRAUMA / INJURY ================
    # ======================================================
    "chōṭ":     ["chot", "chott", "injury", "injured", "zakham", "zakhm",
                 "zakhmi", "wound", "wounded", "ghao", "ghaao",
                 "trauma", "bruise", "bruised"],
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
                 "convulsion", "convulsions", "jhatka", "jhatke", "mirgi", "mirgii",
                 "epilepsy", "epileptic"],
    "falij":    ["faalij", "falaj", "paralysis", "paralyzed", "stroke",
                 "strok", "lakwa", "laqwa", "paralysed", "hemiplegia"],

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
                 "dane", "allergy", "alergy", "reaction", "chubhan",
                 "hives", "urticaria"],

    # ======================================================
    # ============ ADDED: GYNE / OBSTETRIC =================
    # ======================================================
    "haml":     ["hamal", "pregnant", "pregnancy", "pregnent", "delivery",
                 "deliver", "labour", "labor", "zachgi", "zachagi", "miscarriage"],

    # ======================================================
    # ================== ADDED: PSYCH ======================
    # ======================================================
    "ghabrahat":["ghabrahit", "ghabrana", "anxiety", "anxious", "panic",
                 "ghabra", "bechaini", "bechainii",
                 "nervous", "restless", "restlessness"],

    # ======================================================
    # ADDED: CARDIAC QUALITY DESCRIPTORS
    #
    # How the chest feels is the classic discriminator between cardiac and
    # non-cardiac chest pain, and every one of these was reaching the encoder
    # unmapped - "tightness" (384 rows) and "jakar" (350) are the same
    # complaint in two languages, and "bhaari"/"bhari" are the same WORD in
    # two spellings. Kept as three separate concepts rather than one, because
    # tightness, pressure and heaviness are clinically distinct sensations.
    # ======================================================
    "jákṛan":   ["tightness", "tight", "tightening", "jakar", "jakran",
                 "jakarna", "jakdan", "kasav", "kasao", "squeezing",
                 "constriction", "constricting"],
    "dabāo":    ["pressure", "dabao", "dabav", "dabaw", "dabaav", "dabaao",
                 "compression", "compressing"],
    "bhārī":    ["bhaari", "bhari", "bhaaree", "bojh", "boojh",
                 "bojhal", "heaviness", "weight"],
    "tēkha":    ["teekhi", "teekha", "tikhi", "sharp", "stabbing", "shooting",
                 "piercing", "knifelike"],
    "phailna":  ["radiating", "radiate", "radiates", "radiation", "phail",
                 "phailta", "phailti", "spreading", "spreads"],

    # ======================================================
    # ADDED: EXERTION vs REST CONTEXT
    #
    # Exertional versus rest onset is one of the strongest cardiac triage
    # signals there is (rest angina outranks exertional angina). The dataset
    # says it in both languages - "seedhiyan chadhte" (climbing stairs, 746
    # rows) and "exercise" (693) mean the same thing here, as do "aram
    # karte hue" (888) and "sotay" (714) for the rest side.
    # ======================================================
    "mehnat":   ["exercise", "exertion", "exert", "exertional", "seedhiyan",
                 "seerhiyan", "seerhian", "chadhte", "chadhna", "chadhtay",
                 "kaam", "workout", "jogging", "chalte", "chalna", "walking"],
    "ārām":     ["aram", "araam", "resting", "sotay", "sote", "sona", "soté",
                 "letay", "letnay", "lying", "baithe", "baithay", "sitting"],

    # ======================================================
    # ADDED: SEVERITY / ONSET / HISTORY
    # ======================================================
    "shadīd":   ["shadeed", "shadid", "severe", "intense", "unbearable",
                 "extreme", "excruciating"],
    "halkā":    ["halka", "halki", "halke", "mamuli", "mamooli", "maamuli",
                 "mild", "minor", "slight"],
    "achānak":  ["achanak", "acchaanak", "achaanak", "sudden", "suddenly",
                 "abrupt", "abruptly"],
    "purānā":   ["purani", "purana", "puranay", "chronic", "longstanding",
                 "recurring", "recurrent"],
    "mareez":   ["patient", "patients", "mareezh", "mariz", "marizh", "mareeza"],
    "ehsās":    ["ehsaas", "ehsas", "sensation", "feeling", "ajeeb", "strange",
                 "odd", "weird"],
    "thanḍa":   ["thanday", "thande", "thandi", "thandee"],

    # ======================================================
    # ADDED: CARDIAC HISTORY / PROCEDURES
    #
    # Kept as distinct tokens: a prior bypass, a stent and a diagnosis of
    # hypertension are different pieces of history and collapsing them into
    # one "cardiac history" token would throw away the distinction.
    # ======================================================
    "hypertension": ["bloodpressure", "highbp", "hypertensive", "htn"],
    "bypass":   ["bypaas", "cabg", "graft"],
    "stent":    ["stunt", "stents", "angioplasty", "stenting"],
    "surgery":  ["operation", "sarjari", "opration", "surgeries", "procedure"],
    "attack":   ["heartattack", "infarction", "myocardial", "infarct"],
    "irregular":["betarteeb", "beqaida", "arrhythmic", "erratic", "uneven"],

    # ======================================================
    # ADDED: TRIGGER / TIMING / LATERALITY CONTEXT
    #
    # Cardiac triage reads heavily off WHEN a complaint started, WHAT set it
    # off and WHICH side it sits on - "left arm" is not the same complaint as
    # "right arm". All of it was arriving at the encoder in whichever language
    # the nurse happened to type.
    # ======================================================
    "khānā":    ["khana", "khane", "khanay", "khaya", "food", "eating",
                 "meal", "meals", "eaten", "khaana"],
    "subah":    ["morning", "sub", "subha", "savera"],
    "rāat":     ["raat", "night", "nights", "raatko", "raaton"],
    "din":      ["day", "days", "roz", "dino", "dinon", "dn"],
    "bāyāñ":    ["left", "baayan", "bayen", "baen", "baaen", "baaein"],
    "dāyāñ":    ["right", "daayan", "dayen", "daen", "daaen"],
    "taraf":    ["side", "sides", "janib", "taraff"],
    "takleef":  ["taklif", "takliif", "discomfort", "distress", "trouble",
                 "uneasiness", "uneasy"],
    "khāndān":  ["family", "khandani", "khandan", "familial", "hereditary"],
    "tāreekh":  ["history", "maazi", "previous", "prior", "past"],
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

# ============================================================
# FUNCTION-WORD SPELLING VARIANTS
#
# Pure spelling variants of words the stop-word learner ALREADY
# selected. These are grammatical filler, not medical vocabulary,
# so they belong here rather than in DIACRITIZATION_MAP.
#
# THE BUG THIS FIXES: "hai" is a learned stop word and is stripped
# correctly, but the equally common spelling "hay" was never mapped
# back to it, so it survived every stage and reached the encoder.
#
# WHY THIS IS A SEPARATE DICT FROM RULE_REPLACEMENTS: that one is
# applied with a naked str.replace(), i.e. substring matching, which
# is safe for the multi-word phrases it holds but destroys short
# function words. In the 10k corpus "hay" occurs as a substring of
# "kandhay" (shoulders - a body part that must reach DIACRITIZATION_MAP
# intact), "khaya" and "baithay"; a substring rule would silently
# rewrite those to "kandhai"/"khaia"/"baithai". Entries here are
# therefore matched as WHOLE TOKENS only.
# ============================================================

FUNCTION_WORD_VARIANTS = {
    # -> "hai" (learned stop word, 2755 occurrences in the corpus)
    "hay": "hai",
    "hae": "hai",
    # -> "mein" (learned stop word, 4262 occurrences)
    "may": "mein",
    "mai": "mein",
}


def apply_function_word_variants(text):
    """Map function-word spelling variants onto the learned spelling.

    Whole-token replacement only - see FUNCTION_WORD_VARIANTS for why
    substring matching is unsafe here.
    """
    return ' '.join(FUNCTION_WORD_VARIANTS.get(w, w) for w in text.split())


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
    "bhaari pan":        "bhaari",
    "bhari pan":         "bhaari",
}

# ============================================================
# TEXT NORMALIZATION  (normalization -> fuzzy -> diacritization)
# ============================================================

# ============================================================
# FUZZY MATCHING PROTECTION LIST
#
# THE BUG THIS FIXES: RapidFuzz was rewriting ordinary Roman Urdu and
# English function words into medical concepts the patient never
# mentioned. Measured on cardiac_multilingual_10000_v3.csv, 3,373 rows
# (33.7%) carried at least one phantom symptom:
#
#     par   (on/upon)      -> pair   -> pāir     leg        1276 rows
#     hoon  (I am)         -> khoon  -> khūn     blood       274
#     kaha  (said/where)   -> kandha -> kāndha   shoulder    245
#     chaar (four)         -> chakkar-> chákkar  dizziness   230
#     bhar  (full)         -> bukhar -> bukhār   fever       191
#     pan   (in bhaari pan)-> pain   -> dárd     pain        135
#     das   (ten)          -> dast   -> dast     diarrhoea   120
#
# Phantom legs outnumbered genuine "pair" mentions 1276 to 111, so the
# model's leg feature was mostly noise.
#
# Tuning cannot fix this. The false positives score 80.0-88.9 against
# CANONICAL_VOCAB while the one CORRECT rescue in the corpus ("seene" ->
# "seena") scores 80.0 - the lowest of the set - so no threshold
# separates them, and a minimum-length rule that skipped the bad ones
# would skip the good one too. The only honest fix is to name the words
# that must never be guessed at, the same way stopwords.py names the
# clinical vocabulary its statistics must not remove.
#
# Words that genuinely belong to a canonical concept are handled by
# adding them to DIACRITIZATION_MAP instead ("seene", "beat"), which is
# an exact lookup and skips fuzzy matching anyway.
# ============================================================

FUZZY_PROTECTED = {
    # postpositions, conjunctions, particles
    "par", "se", "ka", "ki", "ke", "ko", "mein", "main", "aur", "ya", "to",
    "bhi", "hi", "na", "nahi", "tak", "sath", "saath", "bina", "bagair",
    "jab", "tab", "ab", "phir", "magar", "lekin", "kyunke", "wala", "wali",
    # verb forms
    "hoon", "hain", "hai", "tha", "thi", "the", "raha", "rahi", "rahe",
    "gaya", "gayi", "gaye", "karta", "karti", "karte", "hota", "hoti",
    "hote", "jata", "jati", "jate", "laga", "lagi", "lage", "lagta",
    "lagti", "hua", "hui", "huay", "hue", "chuka", "chuki", "lena", "lene",
    "dena", "dene", "kar", "karna", "hona", "jana",
    # numbers - "chaar" became dizziness, "das" became diarrhoea
    "ek", "do", "teen", "chaar", "char", "panch", "chhe", "saat", "aath",
    "nau", "das", "gyara", "barah", "bees", "pachaas", "sau", "aadha",
    "aadhay", "dono",
    # question words and pronouns
    "kaha", "kahan", "kya", "kyun", "kaise", "kab", "kaun", "mera", "meri",
    "apna", "apni", "uska", "uski", "mujhe", "mujhko", "usko", "unko",
    # time and quantity
    "kal", "aaj", "shaam", "waqt", "ghante", "ghanta", "minute", "roz",
    "hafte", "hafta", "mahine", "mahina", "saal", "baar", "dafa", "der",
    "zyada", "kam", "thora", "thori", "bohat", "bahut", "sara", "sari",
    # ordinary nouns that collided with medical terms
    "bhar", "pan", "kaam", "ghar", "log", "baat", "cheez", "tarah",
    "jaisa", "jaisi", "khud", "sab", "koi", "kuch",
    # English function words
    "on", "in", "at", "of", "and", "or", "but", "the", "is", "was", "are",
    "has", "have", "had", "with", "from", "after", "before", "since",
    "when", "while", "for", "this", "that", "not", "no", "yes", "also",
    "very", "some", "any", "all", "one", "two", "ten",
}


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
    if word in FUZZY_PROTECTED:
        return word
    match, score, _ = process.extractOne(word, CANONICAL_VOCAB, scorer=fuzz.token_sort_ratio)
    return match if score >= threshold else word


def fuzzy_correct_text(text):
    return ' '.join(fuzzy_correct_word(w) for w in text.split())


def apply_diacritization(text):
    """Map every spelling variant to its canonical diacritized form.

    Collapses a stutter created BY this mapping: when two adjacent but
    DIFFERENT source words resolve to the same canonical token, the token
    is emitted once.

    THE BUG THIS FIXES: "seedhiyan chadhte waqt" (climbing stairs) became
    "mehnat mehnat waqt", because "seedhiyan" and "chadhte" are both
    listed under the exertion concept. This is not cosmetic. The
    Bag-of-Words block COUNTS tokens and then multiplies them by the
    attention weights, so a stutter silently doubles that concept's
    contribution for the affected rows and biases the model toward
    whichever concepts happen to own a colliding pair. Measured on
    cardiac_multilingual_10000_v3.csv: 493 rows (4.9%) contained a
    stutter, dominated by "khana khane" -> "khānā khānā" (309 rows) and
    "seedhiyan chadhte" -> "mehnat mehnat" (180).

    Only mappings collapse. A word genuinely repeated in the source is
    left alone, because Urdu reduplicates for meaning - "rukk rukk kar
    dard" is intermittent pain and "kabhi kabhi" is occasionally, and
    flattening those would destroy information rather than restore it.
    The rule is therefore "same canonical, different source word", not
    "same canonical".
    """
    out = []
    prev_src = prev_canon = None
    for word in text.split():
        canon = VARIANT_TO_CANONICAL.get(word, word)
        if canon == prev_canon and word != prev_src:
            prev_src = word          # collision - fold into the previous token
            continue
        out.append(canon)
        prev_src, prev_canon = word, canon
    return ' '.join(out)


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
    """High-confidence fixes applied before fuzzy matching.

    This is a REAL pipeline stage - it is what turns "saans phulna" into
    "saans phoolna", "thanda pasina" into "sweating" and "sugar level"
    into "sugar". Anything that shows the pipeline to a human must show
    this stage too, or words appear to vanish later for no visible reason.

    Two passes, deliberately different in kind:
      1. RULE_REPLACEMENTS      - substring, for multi-word phrases.
      2. FUNCTION_WORD_VARIANTS - whole token, for filler spellings
         ("hay" -> "hai"). Runs second so the phrase rules still see the
         text exactly as they always have, and whole-token only so it
         cannot chew through words like "kandhay" on the way past.
    """
    for k, v in RULE_REPLACEMENTS.items():
        text = text.replace(k, v)
    return apply_function_word_variants(text)


#: The normalization pipeline as an ordered list of (key, title, function).
#: normalize_roman_urdu() and every explain-the-pipeline view are both built
#: from this list, so a stage can never be shown that is not actually run,
#: and a stage can never be run that is not shown.
NORMALIZATION_STAGES = [
    ("clean", "Lowercase + punctuation stripped", clean_text),
    ("rules", "Rule-based phrase + function-word replacements",
     apply_rule_replacements),
    ("fuzzy", "Fuzzy spelling correction", fuzzy_correct_text),
    ("diacritize", "Diacritization", apply_diacritization),
]

STAGE_NOTES = {
    "clean": "everything except a-z becomes a space",
    "rules": ("fixed multi-word phrases from RULE_REPLACEMENTS "
              "(saans phulna -> saans phoolna, sugar level -> sugar), then "
              "whole-token filler spellings from FUNCTION_WORD_VARIANTS "
              "(hay -> hai, may -> mein) so the stop-word stage can match them"),
    "fuzzy": ("RapidFuzz maps an unknown spelling onto the closest word in "
              "CANONICAL_VOCAB at >=80% similarity. It only fires on spellings "
              "the dictionary in the next stage does not already know, so on "
              "many complaints it correctly changes nothing."),
    "diacritize": ("every spelling variant listed in DIACRITIZATION_MAP "
                   "collapses to one canonical form"),
    "stopwords": ("Contribution 1 - the list was learned from the data. A "
                  "token is dropped when it is common AND shows no meaningful "
                  "association with the triage level: near-zero mutual "
                  "information AND a Cramer's V effect size at or below the "
                  "threshold. Effect size, not the chi-square p-value, makes "
                  "the call - p-values shrink as the corpus grows and were "
                  "keeping filler like \"hai\" and \"mein\" in."),
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
    path = os.path.join(resolve_project_file(model_dir), MANIFEST_FILE)
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
        return resolve_project_file(model_dir), ''

    # resolve_project_file() is what makes this work from any working
    # directory: without it the embedding bundle "was not found" whenever
    # the caller happened to be standing somewhere else, and every such
    # run silently downgraded itself to the dictionary model.
    candidate = resolve_project_file(EMBEDDING_MODEL_DIR)
    if not os.path.exists(os.path.join(candidate, 'model.pkl')):
        return resolve_project_file(DICTIONARY_MODEL_DIR), (
            f"'{EMBEDDING_MODEL_DIR}/' not found - using the dictionary model. "
            "Run: python train_embedding_pipeline.py")

    info = describe_model(candidate)
    if info['uses_embeddings'] and allow_fallback:
        try:
            import sentence_transformers          # noqa: F401
        except ImportError:
            return resolve_project_file(DICTIONARY_MODEL_DIR), (
                f"'{candidate}/' needs sentence-transformers "
                f"({info['embedding_model']}), which is not installed - "
                "falling back to the dictionary model in "
                f"'{DICTIONARY_MODEL_DIR}/'. Install it with: "
                "pip install -r requirements-embedding.txt")
    return candidate, ''


def load_artifacts(model_dir='triage_model'):
    """Load the trained model, vectorizers, scaler, encoders and manifest."""
    model_dir = resolve_project_file(model_dir)

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


def load_sentence_transformer(name):
    """Load a sentence-transformer, preferring the local cache.

    THE PROBLEM THIS SOLVES: SentenceTransformer(name) contacts huggingface.co
    on EVERY load to revalidate the snapshot, even when the model is fully
    cached. This project's whole claim is that it runs offline on a hospital
    machine, and on a slow, captive-portal or firewalled network that call does
    not fail fast - it hangs. Measured on this machine: 8 seconds with the hub
    disabled versus over 5 minutes with it reachable-but-slow, during which the
    GUI sits on "Loading model and encoders..." with no way to tell whether it
    is working.

    So: try the cache first and only reach for the network when the model is
    genuinely not present yet (the documented one-time download).
    """
    from sentence_transformers import SentenceTransformer
    try:
        return SentenceTransformer(name, device='cpu', local_files_only=True)
    except Exception:
        # Not cached yet - this is the first run, which is allowed to download.
        return SentenceTransformer(name, device='cpu')


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
        import sentence_transformers          # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            f"The deployed model in '{art['model_dir']}' uses sentence "
            f"embeddings ({name}), so 'sentence-transformers' is required.\n"
            "Install it once (needs internet), then re-run:\n"
            "    pip install -r requirements-embedding.txt"
        ) from e
    art['encoder'] = load_sentence_transformer(name)
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

#: Encoder keys for the four categorical inputs, in the column order the
#: structured block is built in. Training and inference both read this.
CATEGORICAL_ENCODER_KEYS = ['le_gender', 'le_mode', 'le_avpu', 'le_ecg']

# ======================================================================
# NO-TEXT GUARD
#
# THE BUG THIS FIXES: an empty complaint - or one that is only digits or
# only punctuation, both of which clean to the empty string - was
# returning "Level 4, NON-URGENT, 91.2% confidence". That is the single
# worst direction this system can fail in: a nurse who tabs past the
# complaint box, or a CSV whose text column did not survive an export,
# gets a confident DISCHARGE-tier answer. The GUI asked for a complaint
# before predicting, but predict_one() and the whole batch path did not,
# so any file with a blank complaint column went straight through.
#
# WHY FLAG-AND-CAP RATHER THAN A VITALS-ONLY FALLBACK: there is no
# vitals-only model in the bundle, and building one today would mean
# either training a second classifier or feeding the deployed one a
# zeroed text block. A zeroed block is not something the model ever saw
# in training, so it would answer just as confidently from an input it
# has no basis for - trading a visible failure for an invisible one. The
# honest option is to return what the model actually said, refuse to
# dress it up as confident, and make the reason impossible to miss in
# BOTH the single and batch paths.
#
# Note this guard deliberately does NOT catch gibberish ("asdkfj
# qwoeiru"), which survives cleaning as real tokens and still yields a
# confident answer. That is a separate, documented limitation - see
# SUBMISSION_SUMMARY.md - not something a token count can detect.
# ======================================================================

#: Ceiling on reported confidence when the complaint carried no tokens.
MAX_CONFIDENCE_WITHOUT_TEXT = 0.50

NO_TEXT_SIGNAL_WARNING = (
    "NO COMPLAINT TEXT: nothing usable survived cleaning, so this level "
    "comes from the vitals alone and the text features are meaningless. "
    "Confidence capped. Do NOT act on this triage level - re-enter the "
    "complaint."
)


def has_text_signal(text):
    """True when any token survives the normalization pipeline.

    Empty strings, whitespace, digit-only and punctuation-only inputs all
    clean to nothing (clean_text keeps only a-z and whitespace) and
    therefore return False.
    """
    try:
        return bool(normalize_roman_urdu(text).split())
    except Exception:
        return False


def encode_categoricals(art, codes):
    """Turn integer category codes into the layout the model was trained on.

    THE BUG THIS FIXES: the categoricals were fed to the model as the raw
    LabelEncoder integers, i.e. ECG_Status became 0..N in ALPHABETICAL order.
    A linear model reads those codes as magnitudes, so "ST elevation" (10)
    was treated as ten times "Abnormal" (0) - meaningless, and ECG_Status is
    the single most informative feature in the dataset (51.7% of RandomForest
    importance). Measured on cardiac_multilingual_10000.csv, structured
    features alone: 94.5% accuracy / 2.2% under-triage with the ordinal
    codes versus 98.3% / 0.5% one-hot. Under-triage is the number that
    matters clinically, and it drops four-fold.

    Bundles record their choice in the manifest, so the model bundles saved
    before this change (triage_model_embedding_v1_1204rows/ and friends)
    keep loading and predicting exactly as they always did.
    """
    codes = np.asarray(codes, dtype=int)
    if art['manifest'].get('categorical_encoding') != 'onehot':
        return codes.astype(float)          # legacy bundles: ordinal codes
    blocks = []
    for j, key in enumerate(CATEGORICAL_ENCODER_KEYS):
        width = len(art[key].classes_)
        block = np.zeros((codes.shape[0], width))
        # safe_encode already maps unknown categories onto a valid code;
        # clip guards against a bundle whose encoder was replaced by hand.
        block[np.arange(codes.shape[0]), np.clip(codes[:, j], 0, width - 1)] = 1.0
        blocks.append(block)
    return np.hstack(blocks)


def predict_one(art, complaint, age, heart_rate, systolic_bp, diastolic_bp,
                temperature, spo2, gender, mode_of_arrival, avpu, ecg_status,
                warnings=None):
    """Predict triage for a single patient. Returns (level, confidence, proba).

    Pass a list as `warnings` to receive input-quality notes (unknown
    category fallbacks, missing complaint text). The parameter is optional
    so existing three-value callers keep working unchanged, but anything
    user-facing should pass one - the confidence cap alone does not
    explain WHY a number is low.
    """
    text_feat = build_text_features(art, [complaint])

    import pandas as pd
    numerical = art['scaler'].transform(pd.DataFrame(
        [[age, heart_rate, systolic_bp, diastolic_bp, temperature, spo2]],
        columns=NUMERICAL_FEATURES))
    # Unknown-category fallbacks were previously computed and thrown away
    # here; only the batch path ever surfaced them. Collect them so a
    # single prediction can say "ECG_Status was not recognised" too.
    cat_warns = []
    categorical = encode_categoricals(art, [[
        safe_encode(art['le_gender'], gender,         cat_warns, 'Gender'),
        safe_encode(art['le_mode'],   mode_of_arrival, cat_warns, 'Mode_of_Arrival'),
        safe_encode(art['le_avpu'],   avpu,            cat_warns, 'AVPU'),
        safe_encode(art['le_ecg'],    ecg_status,      cat_warns, 'ECG_Status'),
    ]])

    X = np.hstack([numerical, categorical, text_feat])
    proba = art['model'].predict_proba(X)[0]
    level = int(np.argmax(proba))
    confidence = float(proba[level])

    if not has_text_signal(complaint):
        cat_warns.insert(0, NO_TEXT_SIGNAL_WARNING)
        confidence = min(confidence, MAX_CONFIDENCE_WITHOUT_TEXT)

    if warnings is not None:
        warnings.extend(cat_warns)
    return level, confidence, proba


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

    # A header-only sheet is a real thing an operator uploads (they exported
    # the template and saved it before typing anyone in). StandardScaler
    # rejects a 0-row array with "Found array with 0 sample(s)", which surfaced
    # as a raw sklearn traceback in the GUI's error dialog. Return the empty
    # result frame with the full output schema instead, so the caller's
    # "0 patients triaged" path works and the columns are still there.
    if len(df) == 0:
        out = df.copy()
        out['Predicted_Level_0to3'] = pd.Series(dtype=int)
        out['Predicted_Triage_Level'] = pd.Series(dtype=int)
        for col in ['Predicted_Label', 'Confidence', 'P_L0', 'P_L1',
                    'P_L2', 'P_L3', 'Notes']:
            out[col] = pd.Series(dtype=object)
        return out, []

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
    X = np.hstack([num_matrix, encode_categoricals(art, cat_matrix), text_feat])
    proba = art['model'].predict_proba(X)
    levels = np.argmax(proba, axis=1)

    # Same no-text guard as predict_one, applied per row: a blank complaint
    # column in a spreadsheet was the likeliest way to hit this, and it
    # produced a confident NON-URGENT for every such row.
    no_text = [not has_text_signal(t)
               for t in df['Complaint_Text'].fillna('').astype(str)]
    for i, blank in enumerate(no_text):
        if blank:
            row_notes[i] = NO_TEXT_SIGNAL_WARNING + "; " + row_notes[i]

    out = df.copy()
    out['Predicted_Level_0to3']   = levels.astype(int)
    out['Predicted_Triage_Level'] = (levels + 1).astype(int)
    out['Predicted_Label']        = [TRIAGE_LABELS[int(l)].split('(')[0].strip()
                                     for l in levels]
    out['Confidence']             = [
        f"{min(proba[i, levels[i]], MAX_CONFIDENCE_WITHOUT_TEXT if no_text[i] else 1.0)*100:.1f}%"
        + ("  (capped - no complaint text)" if no_text[i] else "")
        for i in range(len(df))]
    # One column per class the MODEL actually has. Hardcoding 4 raised
    # IndexError the moment a dataset carried only 3 triage levels
    # (cardiac_multilingual_10000.csv has no Level 4 at all).
    for k in range(proba.shape[1]):
        out[f'P_L{k}'] = [f"{proba[i, k]*100:.1f}%" for i in range(len(df))]
    out['Notes'] = [n.strip().rstrip(';') for n in row_notes]

    return out, row_notes
