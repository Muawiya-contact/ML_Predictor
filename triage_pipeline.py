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
import numpy as np
import joblib
from rapidfuzz import process, fuzz

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
    """Replace a word with the closest canonical term if score >= threshold."""
    if len(word) < 3:
        return word
    match, score, _ = process.extractOne(word, CANONICAL_VOCAB, scorer=fuzz.token_sort_ratio)
    return match if score >= threshold else word


def fuzzy_correct_text(text):
    return ' '.join(fuzzy_correct_word(w) for w in text.split())


def apply_diacritization(text):
    """Map every spelling variant to its canonical diacritized form."""
    return ' '.join(VARIANT_TO_CANONICAL.get(w, w) for w in text.split())


def normalize_roman_urdu(text):
    """Full text pipeline: clean -> rule replace -> fuzzy -> diacritize."""
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    for k, v in RULE_REPLACEMENTS.items():
        text = text.replace(k, v)
    text = fuzzy_correct_text(text)
    text = apply_diacritization(text)
    return text


# Backwards-compatible alias used by the interactive script
normalize = normalize_roman_urdu


# ============================================================
# ATTENTION
# ============================================================

def build_attention_weights(feature_names):
    """Return the per-feature weight vector for a list of BoW feature names."""
    weights = np.ones(len(feature_names))
    for i, feat in enumerate(feature_names):
        for k, w in MEDICAL_WEIGHTS.items():
            if k in feat:
                weights[i] = w
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


def load_artifacts(model_dir='triage_model'):
    """Load the trained model, vectorizers, scaler and encoders."""
    def p(name):
        return os.path.join(model_dir, name)

    artifacts = {
        'model':     joblib.load(p('model.pkl')),
        'word_bow':  joblib.load(p('word_bow.pkl')),
        'char_bow':  joblib.load(p('char_bow.pkl')),
        'scaler':    joblib.load(p('scaler.pkl')),
        'le_gender': joblib.load(p('gender_enc.pkl')),
        'le_mode':   joblib.load(p('mode_enc.pkl')),
        'le_avpu':   joblib.load(p('avpu_enc.pkl')),
        'le_ecg':    joblib.load(p('ecg_enc.pkl')),
    }
    artifacts['feature_names'] = (
        list(artifacts['word_bow'].get_feature_names_out()) +
        list(artifacts['char_bow'].get_feature_names_out())
    )
    artifacts['attention'] = build_attention_weights(artifacts['feature_names'])
    return artifacts


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
    try:
        f = float(value)
        if np.isnan(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


# ============================================================
# PREDICTION  (single + vectorized batch)
# ============================================================

def predict_one(art, complaint, age, heart_rate, systolic_bp, diastolic_bp,
                temperature, spo2, gender, mode_of_arrival, avpu, ecg_status):
    """Predict triage for a single patient. Returns (level, confidence, proba)."""
    cleaned = normalize_roman_urdu(complaint)
    word_feat = art['word_bow'].transform([cleaned]).toarray()
    char_feat = art['char_bow'].transform([cleaned]).toarray()
    text_feat = np.hstack([word_feat, char_feat]) * art['attention']

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

    num_matrix = np.zeros((len(df), len(NUMERICAL_FEATURES)))
    for j, f in enumerate(NUMERICAL_FEATURES):
        for i, raw in enumerate(df[f].tolist()):
            val = _safe_float(raw, means[f])
            if val != raw and (raw is None or (isinstance(raw, float) and np.isnan(raw))):
                row_notes[i] += f"{f} missing->mean; "
            num_matrix[i, j] = val
    num_matrix = art['scaler'].transform(
        pd.DataFrame(num_matrix, columns=NUMERICAL_FEATURES))

    # --- text features (vectorize the whole column at once) ---
    cleaned = df['Complaint_Text'].fillna('unknown').astype(str).apply(normalize_roman_urdu)
    word_feat = art['word_bow'].transform(cleaned).toarray()
    char_feat = art['char_bow'].transform(cleaned).toarray()
    text_feat = np.hstack([word_feat, char_feat]) * art['attention']

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
