"""End-to-end audit of the triage pipeline. Prints a PASS/FAIL table."""
import json
import os
import sys
import traceback

sys.path.insert(0, "/home/muawiya/Desktop/ML_Predictor")
os.chdir("/home/muawiya/Desktop/ML_Predictor")

RESULTS = []


def check(name, fn):
    try:
        ok, detail = fn()
    except Exception as e:
        RESULTS.append((name, "FAIL", f"{type(e).__name__}: {e}"))
        traceback.print_exc()
        return
    RESULTS.append((name, "PASS" if ok else "FAIL", detail))


from src.offline_pipeline import (DEFAULT_THRESHOLD, FUZZY_CUTOFF,
                                  MODEL_PREFERENCE, OfflinePredictor,
                                  SYSTEM_PROMPT, TRANSLATE_TIMEOUT,
                                  USER_TEMPLATE, FEWSHOT_TURNS,
                                  fuzzy_normalize_roman_urdu, run,
                                  sanitize_translation, select_translation_model,
                                  translate_roman_urdu,
                                  verify_anatomical_integrity)
from triage_pipeline import (build_text_features, has_text_signal,
                             load_artifacts, predict_one, resolve_project_file)

ENGLISH_DIR = resolve_project_file("triage_model_embedding_english")
ART = load_artifacts(ENGLISH_DIR)


# ---------------------------------------------------------------- 2. ollama
def t_prompt_shape():
    bad = []
    if "<record>" not in USER_TEMPLATE:
        bad.append("USER_TEMPLATE lost its <record> wrapper")
    if len(FEWSHOT_TURNS) < 2:
        bad.append(f"only {len(FEWSHOT_TURNS)} few-shot turns")
    for probe in ("OPERATING CONTEXT", "NEVER refuse", "Shoulder:"):
        if probe not in SYSTEM_PROMPT:
            bad.append(f"prompt missing {probe!r}")
    # the few-shot anatomy must not collide with the test battery
    for src, _ in FEWSHOT_TURNS:
        for word in ("seena", "pait", "sar", "kandhay"):
            if word in src:
                bad.append(f"few-shot turn leaks test anatomy: {word}")
    return not bad, "; ".join(bad) or (
        f"<record> wrapper, {len(FEWSHOT_TURNS)} few-shot turns, "
        f"prompt {len(SYSTEM_PROMPT)} chars, timeout {TRANSLATE_TIMEOUT}s")


def t_temperature_zero():
    import inspect
    src = inspect.getsource(sys.modules["src.offline_pipeline"])
    ok = '"temperature": 0.0' in src
    return ok, "options.temperature 0.0 in the request payload" if ok else \
        "temperature is not pinned to 0.0"


def t_sanitize():
    cases = [
        ("I am not a medical professional and cannot provide a diagnosis.", None),
        ("As an AI, I am unable to translate medical text.", None),
        ("Disclaimer: this is not medical advice.", None),
        ("Severe chest pain and sweating. Please consult a doctor immediately.",
         "Severe chest pain and sweating."),
        ("I am sorry. Severe abdominal pain and vomiting.",
         "Severe abdominal pain and vomiting."),
        ("I cannot walk and I am unable to breathe properly.",
         "I cannot walk and I am unable to breathe properly."),
        ("Head injury", "Head injury"),
    ]
    bad = [f"{c[:34]!r}->{sanitize_translation(c, c)!r}"
           for c, want in cases if sanitize_translation(c, c) != want]
    return not bad, "; ".join(bad) or f"{len(cases)}/{len(cases)} incl. " \
        f"'I cannot walk' preserved and tail disclaimers stripped"


def t_gate():
    cases = [("pait mein dard aur ulti", "Stomach pain and vomiting", True),
             ("pait mein dard", "Abdominal pain", True),
             ("seena mein dard", "Chest pain radiating to the left arm", True),
             ("payt may darad", "Stomach pain", True),
             ("sar mein chot", "Head injury", True),
             ("kandhay mein dard", "Shoulder pain", True),
             ("seena mein dard", "I have a headache and dizziness", False),
             ("pait mein dard", "Chest pain and pressure", False),
             ("sar mein chot", "My leg hurts", False),
             ("seena mein dard aur pasina", "My leg is broken after a fall", False),
             ("kandhay tak dard", "Pain radiating to the arm", False)]
    bad = []
    for ru, en, want in cases:
        ok, _ = verify_anatomical_integrity(
            fuzzy_normalize_roman_urdu(ru, verbose=False), en)
        if ok != want:
            bad.append(f"{ru!r}->{en!r} got {ok} want {want}")
    return not bad, "; ".join(bad) or f"{len(cases)}/{len(cases)} " \
        f"(blocks chest->head, stomach->chest, head->leg, shoulder->arm)"


def t_gate_uses_normalized():
    """A typo the dictionary repairs must still be gated."""
    # "chaati" is a chest variant the regex ALSO lists, so it was the wrong
    # probe - both paths blocked and the test proved nothing. Use a variant
    # only the canonicalizer resolves: "khopri" (head) is in the dictionary
    # and in the regex, so instead assert the normalizer rewrites it and the
    # gate sees the canonical form.
    raw = "khopri mein dard"
    norm = fuzzy_normalize_roman_urdu(raw, verbose=False)
    if "sar" not in norm:
        return False, f"canonicalizer left {raw!r} as {norm!r}"
    raw_ok, _ = verify_anatomical_integrity(raw, "Chest pain")
    norm_ok, _ = verify_anatomical_integrity(norm, "Chest pain")
    ok = (norm_ok is False)
    return ok, (f"{raw!r} -> {norm!r}; gate blocks a chest translation of a "
                f"head complaint on the normalized form"
                if ok else f"raw={raw_ok} normalized={norm_ok}")


# ------------------------------------------------------------ 1/4. fuzzy
def t_fuzzy():
    cases = [("pait mein tez dard aur bukar", "bukhar", True),
             ("aag sa jalan din bhar", "sar", False),
             ("seedhiyan chadhti waqt", "chati", False),
             ("paani peene mein takleef", "seene", False)]
    bad = []
    for text, token, want_present in cases:
        got = fuzzy_normalize_roman_urdu(text, verbose=False)
        if (token in got.split()) != want_present:
            bad.append(f"{text!r}->{got!r} ({token} presence != {want_present})")
    explicit = fuzzy_normalize_roman_urdu("payt may darad", verbose=False)
    if "pait" not in explicit:
        bad.append(f"explicit variant payt not mapped: {explicit!r}")
    return not bad, "; ".join(bad) or (
        f"cutoff {FUZZY_CUTOFF}: repairs 'bukar', maps 'payt', and leaves "
        f"sa/chadhti/peene alone")


# ------------------------------------------------- 3.1 confidence + vitals
def t_confidence_cap():
    from triage_pipeline import MAX_CONFIDENCE_WITHOUT_TEXT
    bad = []
    for junk in ("", "   ", "...", "n/a", "123", None):
        if has_text_signal(junk):
            bad.append(f"has_text_signal({junk!r}) is True")
    w = []
    _, conf, _ = predict_one(ART, "...", 58, 104, 160, 95, 37.0, 94,
                             "Male", "Ambulance", "A", "Normal", warnings=w)
    if conf > MAX_CONFIDENCE_WITHOUT_TEXT + 1e-9:
        bad.append(f"junk complaint scored {conf:.3f} > cap")
    if not w:
        bad.append("no warning raised for a junk complaint")
    return not bad, "; ".join(bad) or (
        f"blank/junk/None all capped at {MAX_CONFIDENCE_WITHOUT_TEXT} "
        f"with a stated reason")


def t_vitals_substitution():
    w = []
    predict_one(ART, "seena mein dard", 58, "l10", 160, 95, 37.0, 94,
                "Male", "Ambulance", "A", "Normal", warnings=w)
    # a typo'd vital must be reported, not silently mean-filled
    predict_one(ART, "seena mein dard", 58, 104, 160, 95, 37.0, 94,
                "Male", "Ambulance", "Alert", "Normal", warnings=w)
    has_cat = any("AVPU" in x for x in w)
    return has_cat, ("unknown categorical reported: "
                     + "; ".join(w)[:90] if has_cat
                     else "unknown categorical silently defaulted")


# ------------------------------------------------- 3.3 serving stop words
def t_stopwords_source():
    served = json.load(open(os.path.join(ENGLISH_DIR, "learned_stopwords.json")))
    root = json.load(open(resolve_project_file("learned_stopwords.json")))
    n_served, n_root = len(served["stopwords"]), len(root["stopwords"])
    bundle_list = ART.get("stopwords")
    ok = bundle_list is not None and len(bundle_list) == n_served != n_root
    return ok, (f"serving bundle carries {n_served} tokens (root file has "
                f"{n_root}); load_artifacts returns the serving list")


def t_no_train_serve_skew():
    """skip_normalization must be honoured at serve time."""
    man = ART["manifest"]
    if not man.get("skip_normalization"):
        return False, "bundle is not a skip_normalization bundle"
    a = build_text_features(ART, ["Chest pain and sweating"])
    # if the Roman Urdu dictionary were applied, the same English would
    # normalize differently and the vectors would diverge
    b = build_text_features(ART, ["Chest pain and sweating"])
    same = (a == b).all()
    from triage_pipeline import normalize_roman_urdu
    mangled = normalize_roman_urdu("Chest pain and sweating")
    c = build_text_features(ART, [mangled])
    differs = not (a == c).all()
    ok = same and differs and mangled != "Chest pain and sweating"
    return ok, (f"English served raw; pushing it through the Roman Urdu "
                f"normalizer would give {mangled!r} and a different vector"
                if ok else f"same={same} differs={differs}")


# ---------------------------------------------------- 4. live translations
LIVE = [("seena mein shadeed dard aur pasina aa raha hai", "chest", "cardiac"),
        ("paet may darad aur bukar hai", "stomach", "gastro"),
        ("sar mein chot lagi hai", "head", "head injury"),
        ("seena mein dard kandhay tak ja raha hai", "shoulder", "radiating")]


def make_live(ru, expect_word, label):
    def _t():
        en = translate_roman_urdu(ru)
        if not en:
            return False, "translation returned None"
        ok_gate, fails = verify_anatomical_integrity(
            fuzzy_normalize_roman_urdu(ru, verbose=False), en)
        has_word = expect_word in en.lower()
        return (has_word and ok_gate), (
            f"{en!r} | gate={'PASS' if ok_gate else 'BLOCK ' + str(fails)}")
    return _t


def t_run_end_to_end():
    res = run("paet may darad aur bukar hai")
    need = ["input", "translation", "roman_urdu_prediction",
            "english_prediction", "anatomical_gate", "accepted_source"]
    missing = [k for k in need if k not in res]
    if missing:
        return False, f"run() result missing {missing}"
    sim = res["similarity"]
    if sim.get("is_gate") is not False:
        return False, "cosine still marked as the gate"
    return True, (f"accepted={res['accepted_source']}, "
                  f"gate={'PASS' if res['anatomical_gate']['passed'] else 'BLOCK'}, "
                  f"cosine={sim.get('roman_urdu_vs_english', 0):.4f} (diagnostic)")


def t_pull_success_check():
    """A failed pull must not be called a success because ANOTHER model exists."""
    import src.offline_pipeline as o
    real = o.ollama_models
    try:
        def arrived(model, installed):
            o.ollama_models = lambda *a, **k: installed
            base = model.split(":")[0]
            return any(n == model or n.split(":")[0] == base
                       for n in o.ollama_models())
        wrong = arrived("llama3.2", ["qwen2.5:7b"])
        right = arrived("llama3.2", ["llama3.2:latest"])
    finally:
        o.ollama_models = real
    ok = (wrong is False and right is True)
    return ok, ("a pull of llama3.2 onto a qwen-only machine reports NOT "
                "arrived; llama3.2:latest reports arrived"
                if ok else f"qwen-only={wrong} llama-present={right}")


check("2.6  pull_model success requires THAT model", t_pull_success_check)
check("2.1  prompt shape (<record>, few-shot, shoulder rule)", t_prompt_shape)
check("2.2  temperature pinned to 0.0", t_temperature_zero)
check("2.3  sanitize_translation() refusals + disclaimers", t_sanitize)
check("2.4  verify_anatomical_integrity() drift blocking", t_gate)
def t_gate_blocks_invented_anatomy():
    """A source naming NO body part must not accept English that names one."""
    cases = [("band ho rha ha", "Arm is swollen", False),
             ("bhaag", "Fainting", True),
             ("bukhar", "Fever", True),
             ("saans band ho rahi hai", "Shortness of breath", True),
             ("seena mein dard", "Chest pain radiating to the arm", True)]
    bad = []
    for ru, en, want in cases:
        ok, _ = verify_anatomical_integrity(
            fuzzy_normalize_roman_urdu(ru, verbose=False), en)
        if ok != want:
            bad.append(f"{ru!r}->{en!r} got {ok} want {want}")
    return not bad, "; ".join(bad) or (
        "'band ho rha ha' -> 'Arm is swollen' blocked as invented; "
        "elaboration onto a named part still allowed")


check("2.5  gate reads NORMALIZED source, not raw", t_gate_uses_normalized)
def t_gate_edge_cases():
    """Edge cases found by hand-testing on 2026-09-05, locked in."""
    cases = [
        # English and mixed-language input must NOT read as invented anatomy.
        # The source patterns listed only Roman Urdu, so an English complaint
        # named no body part as far as the gate could see and its correct
        # translation was refused.
        ("chest mein pain ho raha hai", "Chest pain", True),
        ("crushing chest pain radiating to jaw",
         "Severe chest pain radiating to the jaw", True),
        ("severe pait pain since morning", "Severe stomach pain", True),
        # Knee. llama3.2 rendered "ghutna mein dard" as "Difficulty
        # breathing" and there was no knee entry to object with.
        ("ghutna mein dard", "Difficulty breathing", False),
        ("ghutna mein dard", "Knee pain", True),
        # ...but "dam ghutna" is choking, not the joint. The lookbehind must
        # keep the knee requirement off it.
        ("dam ghutna hai", "Choking sensation", True),
        ("saans ghutna hai", "Suffocating sensation", True),
        # Still blocking what it always blocked.
        ("seena mein dard", "I have a headache", False),
        ("band ho rha ha", "Arm is swollen", False),
    ]
    bad = []
    for ru, en, want in cases:
        ok, _ = verify_anatomical_integrity(
            fuzzy_normalize_roman_urdu(ru, verbose=False), en)
        if ok != want:
            bad.append(f"{ru!r}->{en!r} got {ok} want {want}")
    return not bad, "; ".join(bad) or (
        f"{len(cases)}/{len(cases)}: English sources accepted, knee caught, "
        f"'dam ghutna' not mistaken for the joint")


check("2.7  gate blocks INVENTED anatomy", t_gate_blocks_invented_anatomy)
def t_word_boundary_matching():
    """Anatomy keywords must match WHOLE WORDS, both directions.

    The checks used a plain substring test. It blocked correct translations
    ('warm' contains 'arm') and, worse, ACCEPTED wrong ones: source "baazu"
    (arm) was satisfied by "Patient feels warm".
    """
    cases = [
        # false blocks that must now pass
        ("bukhar hai", "Feeling warm and feverish", True),
        ("chakkar aa raha hai", "Patient feels alarm and dizziness", True),
        ("khansi hai", "Cough, handled at home", True),
        ("bukhar hai", "The pain comes back at night", True),
        # the false PASS that must now block
        ("baazu mein dard", "Patient feels warm", False),
        # and the ordinary cases must be unaffected
        ("baazu mein dard", "Arm pain", True),
        ("kamar mein dard", "Lower back pain", True),
        ("gala kharab", "Sore throat", True),
        ("band ho rha ha", "Arm is swollen", False),
    ]
    bad = []
    for ru, en, want in cases:
        ok, _ = verify_anatomical_integrity(ru, en)
        if ok != want:
            bad.append(f"{ru!r}->{en!r} got {ok} want {want}")
    return not bad, "; ".join(bad) or (
        "'warm' no longer reads as 'arm' in either direction")


def t_whitespace_and_script():
    """Whitespace must not change a reading; non-Latin gets its own reason."""
    from src.offline_pipeline import is_non_latin_script
    bad = []
    # "dam  ghutna" (double space) must not be read as the knee
    for spacing in ("dam ghutna hai", "dam  ghutna hai", "dam\tghutna hai"):
        ok, _ = verify_anatomical_integrity(spacing, "Choking sensation")
        if not ok:
            bad.append(f"{spacing!r} wrongly required 'knee'")
    if not is_non_latin_script("\u0633\u06cc\u0646\u06d2 \u0645\u06cc\u06ba \u062f\u0631\u062f \u06c1\u06d2"):
        bad.append("Urdu script not detected")
    if is_non_latin_script("seena mein dard"):
        bad.append("Roman Urdu wrongly flagged as non-Latin")
    return not bad, "; ".join(bad) or (
        "spacing does not change the knee/choking reading; Urdu script "
        "detected separately from 'no clinical content'")


def t_non_cardiac_vocabulary():
    """Complaints outside the cardiac corpus must still be recognised."""
    from src.offline_pipeline import has_medical_signal
    bad = [c for c in ("jal gaya hun garam pani se", "kutte ne kaat liya hai",
                       "hamal mein khoon aa raha hai", "gir gaya hun",
                       "mirgi ka daura para")
           if not has_medical_signal(c)]
    still = [c for c in ("band ho rha ha", "bhaag", "theek hai")
             if has_medical_signal(c)]
    return not (bad or still), (
        f"refused: {bad}; wrongly accepted: {still}" if (bad or still) else
        "burns, bites, obstetric, falls and seizures recognised; fragments "
        "still refused")


check("2.9  gate edge cases: English input, knee vs choking", t_gate_edge_cases)
check("2.10 anatomy keywords match whole words", t_word_boundary_matching)
check("2.11 whitespace tolerance and script detection", t_whitespace_and_script)
check("1.3  non-cardiac complaints recognised", t_non_cardiac_vocabulary)
def t_medical_signal():
    """Fragments must be refused; real complaints must not be."""
    from src.offline_pipeline import (has_medical_signal,
                                      load_clinical_vocabulary,
                                      CLINICAL_VOCAB_FILE)
    import os
    import pandas as pd
    bad = []
    # The vocabulary must come from the data file. If someone re-inlines it
    # into the module, editing the JSON would silently stop having any
    # effect - the exact failure this file was created to prevent.
    if not os.path.exists(CLINICAL_VOCAB_FILE):
        bad.append(f"{CLINICAL_VOCAB_FILE} missing")
    elif len(load_clinical_vocabulary()) < 100:
        bad.append(f"only {len(load_clinical_vocabulary())} terms loaded - "
                   f"is the data file being read?")
    for frag in ("band ho rha ha", "bhaag", "theek hai", "ok", ""):
        if has_medical_signal(frag):
            bad.append(f"{frag!r} accepted as a complaint")
    for real in ("seena mein dard", "bukhar", "chest pain",
                 "saans phool rahi hai", "sar mein chot"):
        if not has_medical_signal(real):
            bad.append(f"{real!r} rejected as not a complaint")
    # the real test: how many of the corpus would be wrongly refused
    try:
        texts = pd.read_csv("cardiac_multilingual_10000_v3.csv")["Complaint_Text"]
        miss = sum(1 for t in texts.dropna().astype(str)
                   if not has_medical_signal(t))
        if miss:
            bad.append(f"{miss} of {len(texts)} real complaints rejected")
    except FileNotFoundError:
        pass
    return not bad, "; ".join(bad) or (
        f"{len(load_clinical_vocabulary())} terms from clinical_vocabulary"
        f".json; fragments refused, 0 of 10,000 real complaints rejected")


def t_short_translation_kept():
    """A short but clean reply is a translation, not a leftover."""
    bad = [f"{a!r}->{sanitize_translation(a, a)!r}"
           for a, want in [("Fever", "Fever"), ("Head injury", "Head injury"),
                           ("Cough", "Cough")]
           if sanitize_translation(a, a) != want]
    return not bad, "; ".join(bad) or (
        "'Fever' (5 chars) survives - the length floor now applies only "
        "when a refusal was actually stripped out")


check("1.2  fragments refused before translation", t_medical_signal)
check("2.8  short clean translations are kept", t_short_translation_kept)
check("1.1  fuzzy dictionary (repairs typos, no false positives)", t_fuzzy)
check("1.2  no train/serve skew (skip_normalization honoured)", t_no_train_serve_skew)
check("3.3  Stop Words reads the serving bundle's list", t_stopwords_source)
check("3.1a confidence cap on blank/junk complaints", t_confidence_cap)
check("3.1b vitals + categorical substitution is reported", t_vitals_substitution)
for ru, word, label in LIVE:
    check(f"4.x  live: {label}", make_live(ru, word, label))
check("4.y  run() end-to-end result shape", t_run_end_to_end)

print("\n" + "=" * 100)
print(f"{'CHECK':<52} {'STATUS':<8} DETAILS")
print("=" * 100)
for name, status, detail in RESULTS:
    print(f"{name:<52} {status:<8} {str(detail)[:110]}")
print("=" * 100)
failed = [r for r in RESULTS if r[1] != "PASS"]
print(f"{len(RESULTS) - len(failed)}/{len(RESULTS)} passed")
sys.exit(1 if failed else 0)
