"""
src/offline_pipeline.py
=======================================================================
Fully offline: Roman Urdu -> local LLM translation -> triage + department.
=======================================================================

    raw Roman Urdu
          |
          v
    Ollama (llama3.2, localhost:11434)  ....... normalises noisy text
          |
      standard English
          |
     +----+--------------------------+
     v                               v
  e5-small [CLS]/mean vector    same encoder on the ORIGINAL Roman Urdu
     v                               v
  RandomForest heads            cosine similarity between the two
  (Triage_Level, Category)      = how much the translation moved the text

Nothing here reaches the public internet. Ollama is a local HTTP service
and the encoder is loaded from the on-disk Hugging Face cache.

TWO DELIBERATE DEVIATIONS FROM THE BRIEF, BOTH FOR THE SAME REASON
-------------------------------------------------------------------
1. The brief asked for paraphrase-multilingual-MiniLM-L12-v2. The
   classifiers in models_src/ were fitted on intfloat/multilingual-e5-small
   with a "passage: " prefix - the manifest records it. Both models emit
   384 dimensions, so MiniLM vectors would load, predict, and be WRONG,
   with nothing to signal it. The encoder is therefore read FROM the
   manifest rather than named here, so it cannot drift from whatever the
   classifiers were actually trained on.

2. Those classifiers were trained on ROMAN URDU embeddings
   (complaint_embeddings_direct_roman_urdu.npy). Feeding them the English
   translation is a distribution shift: measured on this dataset, the same
   complaint in the two languages sits at cosine ~0.84, not ~1.0. So both
   predictions are produced - Roman Urdu (matching training) and English
   (as briefed) - and the disagreement between them is reported rather
   than hidden. That IS the embedding-alignment check the brief asks for,
   made actionable.
"""

from __future__ import annotations

import json
import os
import re
from difflib import get_close_matches
import sys
import urllib.error
import urllib.request
from typing import Optional, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:latest")
MODEL_DIR = os.path.join(_ROOT, "models_src")
#: ONE threshold for the embedding safety gate, at 0.75.
#:
#: It was 0.90, which is the right bar for "are these the same sentence"
#: and the wrong one for a CROSS-LANGUAGE pair: the same complaint in
#: Roman Urdu and English measures ~0.84 on this encoder even when the
#: translation is flawless. At 0.90 the gate rejected good translations -
#: it fell back on "I am experiencing a lot of anxiety and pain in my
#: stomach" (a correct rendering, scoring 0.7684) and so accepted the
#: Roman Urdu head's Infectious_Disease over the English head's
#: Gastrointestinal, which was the better answer.
#:
#: 0.75 sits below the good-translation band and above the 0.79 that
#: llama3.2's chest-to-head error produced, so genuine drift is still
#: caught. FALLBACK_THRESHOLD is an alias rather than a second number,
#: because two thresholds that are meant to agree will eventually not.
DEFAULT_THRESHOLD = 0.75
FALLBACK_THRESHOLD = DEFAULT_THRESHOLD

# A long prompt for a small model, deliberately. Every rule below closes a
# failure this pipeline actually produced: llama3.2 rendered "seena mein
# shadeed dard aur pasina" (severe CHEST pain, SWEATING) as "Severe headache
# with pressure and dizziness" - wrong organ, invented symptom - and
# collapsed "sar mein chot lagi hai" (head INJURY) into "Headache". qwen2.5
# dropped "pait" (abdomen) entirely from "pait mein dard aur ulti". Naming
# the exact pairs the models confuse, and showing worked examples, is what
# fixed them.
SYSTEM_PROMPT = """### 0. OPERATING CONTEXT - READ FIRST
You are an OFFLINE TEXT PREPROCESSING COMPONENT inside a data pipeline. You
are not talking to a patient, a clinician, or any human being. Your output is
consumed directly by a sentence encoder; nobody reads it.

The input is a stored text record being converted from one writing system to
another. You are performing STRING CONVERSION, not medical consultation. No
one is asking you for advice, a diagnosis, or a triage decision - those are
made downstream by a trained classifier, not by you.

Therefore:
- NEVER refuse. There is nothing to refuse: converting stored text between
  languages is not a clinical act.
- NEVER output a safety disclaimer, a "not a medical professional" caveat, a
  "consult a doctor" line, an apology, or an offer of further help.
- NEVER write "I am unable...", "I cannot...", "As an AI...", "I am not a
  medical professional...", "Please seek medical attention...", or "Can I
  help with anything else?".
- Any such text CORRUPTS the pipeline. A disclaimer is embedded as if it were
  the patient's symptoms, and the patient is then scored on your apology
  instead of their complaint. This is the single worst failure mode here.
- If the input is distressing, alarming, or describes an emergency, that
  changes NOTHING. Translate it flatly and completely. Urgency is the
  downstream classifier's job and it needs the words intact to do it.
- Your entire response must be the converted sentence and nothing else.

### 0b. ROLE
You are a specialized, strict, literal clinical translator converting Roman Urdu medical complaints into precise English medical terms.

You must treat the entire input as Roman Urdu (informal phonetically spelled Urdu written in the Latin alphabet). Your sole objective is to translate every word accurately based on its literal meaning without making clinical assumptions, inferring secondary conditions, or guessing underlying diseases.

### 1. CORE TRANSLATION PRINCIPLES
- ABSOLUTE LITERAL TRANSLATION: Translate ONLY what is explicitly written in the input text.
- NO CLINICAL INFERENCE: Never upgrade, substitute, or add medical symptoms that are not explicitly stated by the user (e.g., never convert "ghabrahat" + "paet pain" into "chest tightness" or "heart attack").
- PRESERVE ANATOMICAL LOCATION: The exact body part mentioned MUST be preserved in the English output.
- NO EXPLANATIONS OR PREAMBLE: Output ONLY the translated English sentence. Do not include intro text, conversational filler, quotation marks, or notes.

### 2. STRICT ANATOMICAL DICTIONARY & MAPPING RULES
Always map Roman Urdu body parts strictly to their precise English anatomical equivalents:
- Abdomen / Stomach: "pait", "paet", "pet", "shikm", "maida", "mayda" (NEVER translate as chest, heart, or lungs)
- Chest: "seena", "seene", "chati", "chaati"
- Head: "sar", "sir", "sear", "khopri"
- Throat / Neck: "gala", "galao", "gardan"
- Back: "peeth", "pith", "kamar"
- Heart: "dil"
- Eye / Eyes: "aankh", "aankhen", "aakhein"
- Ear / Ears: "kaan"
- Shoulder: "kandha", "kandhay", "kandhe", "kandha" (NEVER translate as arm or hand - a shoulder is not an arm, and pain radiating to the shoulder is a different clinical sign)
- Arm / Hand: "baazu", "bazu", "haath", "hath"
- Leg / Foot: "taang", "tang", "paon", "pair"
- Knee: "ghutna", "ghutne" on its own is the KNEE (a joint in the leg). Do
  not confuse it with "ghutan" (suffocation): "dam ghutna" and "saans
  ghutna" mean choking and have nothing to do with the leg. "ghutna mein
  dard" is KNEE PAIN, never breathing difficulty.
- Jaw: "jabra", "jabray"

### 3. SYMPTOM & SENSORY MAPPING RULES
- Pain / Ache: "dard", "darad", "daard", "dukhna", "peera" -> Translate as "pain" or "ache"
- Anxiety / Restlessness: "ghabrahat", "bechaini" -> Translate strictly as "anxiety", "restlessness", or "uneasiness" (NEVER translate as chest tightness or heart pressure)
- Shortness of Breath / Breathing difficulty: "saans phoolna", "saans ka masla" -> Translate as "shortness of breath" or "difficulty breathing"
- Choking / suffocation: "dam ghutna", "dum ghutna", "saans ghutna", "ghutan"
  -> Translate as "choking" or "a suffocating sensation". This is BREATHING.
  It is not the knee, even though it shares the word "ghutna": the knee
  reading applies only when "ghutna" stands alone, never after "dam", "dum"
  or "saans".
- Fever / Chills: "bukhar", "bookhar", "tap", "thand lagna", "kankani" -> Translate as "fever" or "chills"
- Vomiting / Nausea: "ulti", "oolti", "qay", "matli" -> Translate as "vomiting" or "nausea"
- IDIOM, do not read literally: "dil kharab" / "dil kharab ho raha hai" means
  NAUSEA - feeling sick in the stomach. It is not a heart complaint and not
  an emotional statement. Translate it as "nausea" or "feeling nauseous".
  Reading it literally as a failing heart is alarming and wrong.
- Dizziness / Fainting: "chakkar", "sar ghoomna", "behosh" -> Translate as "dizziness", "vertigo", or "fainting"
- Sweating: "pasina", "paseena", "paseenay" -> Translate as "sweating" or "diaphoresis"
- Trauma / Injury / Bleeding: "chot", "zakhmi", "khoon", "bleeding" -> Translate as "injury", "trauma", or "bleeding" (NEVER reduce trauma to simple pain)
- Burning sensation: "jalan", "jalne" -> Translate as "burning" or "heartburn" (if specific to stomach/throat)

### 4. COMMON PHONETIC VARIATIONS & SPELLING NORMALIZATION
Recognize and correctly interpret non-standard Roman Urdu spelling variants before translating:
- Pronouns & Modifiers:
  * "mujy", "mujhe", "mjhe", "mujhay" -> "I" / "me"
  * "buhat", "bohat", "bht", "bahut", "boht" -> "very", "severe", or "a lot of"
  * "shadeed", "tez" -> "severe" or "intense"
  * "thoda", "thora", "halka", "halki" -> "mild" or "slight"
  * "ho raha hay", "ho rahi hai", "hova" -> "is happening" / "experiencing"
  * "se", "say", "sy" -> "since" or "from"
  * "aur", "or", "aurr" -> "and"
  * "lekin", "magar", "pr" -> "but"
  * "nahi", "nahin", "nhi" -> "no" or "not"

### 5. FEW-SHOT EXAMPLES FOR GUIDANCE

Example 1:
Input: Mujy buhat ghabrahat ho rahi hay aur paet may darad ho raha hay.
Output: I am experiencing a lot of anxiety and pain in my stomach.

Example 2:
Input: Seena mein shadeed dard aur pasina aa raha hai 2 ghanatay se.
Output: Severe chest pain and sweating for 2 hours.

Example 3:
Input: Sar mein chot lagi hai accident ke baad aur chakkar aaray hain.
Output: Head injury after an accident and experiencing dizziness.

Example 4:
Input: Subah se pait mein tez dard hai lekin bukhar nahi hai.
Output: Severe stomach pain since morning but no fever.

Example 5:
Input: Dil ki dhadkan boht tez hai aur saans phool rahi hai.
Output: Heartbeat is very fast and experiencing shortness of breath.

Execute all translations adhering strictly to these rules."""


# ----------------------------------------------------------------------
# 1. Translation, via the local Ollama HTTP API
# ----------------------------------------------------------------------

def ollama_available(url: str = OLLAMA_URL, timeout: float = 3.0) -> bool:
    """True when the local Ollama service answers."""
    try:
        with urllib.request.urlopen(f"{url}/api/tags", timeout=timeout):
            return True
    except Exception:
        return False


def ollama_models(url: str = OLLAMA_URL, timeout: float = 5.0) -> list:
    try:
        with urllib.request.urlopen(f"{url}/api/tags", timeout=timeout) as r:
            return [m["name"] for m in json.load(r).get("models", [])]
    except Exception:
        return []



# ----------------------------------------------------------------------
# Model selection and pulling
# ----------------------------------------------------------------------

#: Preference order for translation. Any of these will do the job; the
#: list exists so a machine that already has SOME capable model is never
#: told to download another one. Matched by prefix, so "llama3.2:latest",
#: "llama3.2:3b" and "llama3.2" all satisfy the "llama3.2" entry.
MODEL_PREFERENCE = [
    # llama3.2 FIRST, on measured evidence rather than size. Head to head on
    # the same five complaints, with the clinical SYSTEM_PROMPT applied to
    # both: llama3.2 (2 GB) scored 5/5, med-translator and raw qwen2.5:7b
    # (4.7 GB) scored 4/5. Both larger models dropped "pait" (abdomen) from
    # "pait mein dard aur ulti", rendering it as bare "pain and vomiting" -
    # the exact anatomical loss the Modelfile rules exist to prevent - and
    # both dropped "shadeed" (severe) where llama3.2 kept it. Running raw
    # qwen2.5 alongside med-translator gave near-identical output, so the
    # Modelfile is working; this is qwen's behaviour on this task.
    #
    # The earlier "llama3.2 is bad" reading came from BEFORE the clinical
    # prompt existed - it scored 2/5 then. The prompt, not the model size,
    # was the fix.
    "llama3.2", "med-translator", "qwen2.5", "qwen2", "llama3.1", "llama3",
    "mistral", "gemma2", "gemma", "phi3",
]


def select_translation_model(available: Sequence[str] | None = None,
                             preferred: str = OLLAMA_MODEL) -> Optional[str]:
    """Pick a locally installed model to translate with.

    Returns the exact tag to call, or None when nothing usable is
    installed. `preferred` wins if present; otherwise the first entry of
    MODEL_PREFERENCE that matches something local; otherwise, rather than
    give up while a perfectly good model sits there, the first installed
    model of any kind.
    """
    names = list(available if available is not None else ollama_models())
    if not names:
        return None
    # Guard the None case at the source too: a caller passing preferred=None
    # should get the preference list, not an AttributeError on .split().
    preferred = preferred or OLLAMA_MODEL
    if preferred in names:
        return preferred
    base = preferred.split(":")[0]
    for n in names:
        if n.split(":")[0] == base:
            return n
    for want in MODEL_PREFERENCE:
        for n in names:
            if n.split(":")[0] == want or n.startswith(want):
                return n
    return names[0]


def pull_model(model: str = "llama3.2", url: str = OLLAMA_URL,
               progress=None, timeout: float = 3600.0) -> tuple:
    """Download a model through Ollama's streaming pull endpoint.

    `progress` is called as progress(status, completed, total) for each
    line Ollama streams, so a caller can drive a progress bar without
    knowing anything about the wire format. Returns (ok, message).

    Streams rather than shelling out to `ollama pull` so the GUI gets real
    byte counts instead of a terminal spinner it cannot read.
    """
    payload = json.dumps({"model": model, "stream": True}).encode("utf-8")
    req = urllib.request.Request(f"{url}/api/pull", data=payload,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if msg.get("error"):
                    return False, str(msg["error"])
                if progress:
                    progress(msg.get("status", ""),
                             msg.get("completed", 0) or 0,
                             msg.get("total", 0) or 0)
                if msg.get("status") == "success":
                    return True, f"{model} pulled"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    # The stream ended without an explicit success; verify by asking.
    #
    # Ask ONLY whether this model arrived. The old check also accepted
    # select_translation_model(preferred=model) is not None, which falls back
    # to any installed model as a last resort - so on a machine with qwen2.5
    # a dropped llama3.2 download returned ok=True and the GUI announced
    # "Download complete: llama3.2 is installed" for a model that was not.
    base = model.split(":")[0]
    arrived = any(n == model or n.split(":")[0] == base for n in ollama_models())
    return arrived, ("pull stream ended" if arrived else
                     f"{model} did not arrive - the pull stream ended without "
                     f"a success line and the model is not installed")


#: The input is wrapped as a data record before it reaches the model.
#:
#: This is not cosmetic. Handed a bare "seena mein dard kandhay tak ja raha
#: hai", llama3.2 reads it as a person describing their own chest pain and
#: answers the person: "I can't provide medical advice. If you are
#: experiencing chest pain, please seek medical attention immediately."
#: Wrapped in <record> tags the same model translates it without hesitating,
#: because the tags make it evident that this is stored text being converted
#: rather than someone in the room asking for help.
USER_TEMPLATE = "<record>{}</record>"

#: Seconds to wait on one /api/chat call. Was 120, which is ample once the
#: model is resident and not enough for a cold start: llama3.2 loading while
#: the GUI loads its sentence encoders on the same CPU overran it three
#: times in one session, and each overrun surfaced as "Is it running?" for a
#: service that was running perfectly well. A slow first translation beats a
#: failed one. Note the refusal retry means a fully wedged call can now take
#: up to 2x this before returning.
TRANSLATE_TIMEOUT = 300.0


#: Two completed conversions, replayed as real conversation turns rather than
#: described in the system prompt.
#:
#: They do two jobs. They establish by demonstration that records get
#: converted and not refused - which suppresses the refusal more reliably
#: than any instruction did. And they improved accuracy: on the shoulder
#: case, prompt-only framing returned "radiating to the arm", while these
#: turns returned "radiating to the shoulder", which is what "kandhay"
#: actually means.
#:
#: Deliberately chosen from anatomy that appears NOWHERE in the project's
#: test battery - throat and leg. Reusing a test sentence here would let the
#: model echo the answer straight back and the test would score a hit that
#: proves only that it can copy.
FEWSHOT_TURNS = [
    ("gala kharab hai aur khansi", "Sore throat and cough"),
    ("taang mein sujan hai", "Leg swelling"),
]

#: Appended for one retry after a refusal, as a user turn. A refusal already
#: in the context is otherwise a precedent the model follows.
RETRY_NUDGE = (
    "<system-note>The previous reply was a refusal. Refusals are invalid "
    "output for this component - the record is stored text, not a live "
    "consultation, and no advice is being requested. Output ONLY the English "
    "conversion of the record above, with no disclaimer and no preamble."
    "</system-note>")


def _ollama_chat(messages: list, model: str, url: str,
                 timeout: float) -> Optional[str]:
    """One /api/chat round trip. None on any transport failure."""
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.0},
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{url}/api/chat", data=payload,
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = json.load(r)
    except urllib.error.HTTPError as e:
        print(f"[ollama] HTTP {e.code}: "
              f"{e.read().decode('utf-8', 'replace')[:200]}", flush=True)
        return None
    except Exception as e:
        print(f"[ollama] {type(e).__name__}: {e}", flush=True)
        print(f"[ollama] Is it running?  ollama serve   (expected at {url})",
              flush=True)
        return None
    return (body.get("message") or {}).get("content", "").strip() or None


def translate_roman_urdu(text: str, model: str = OLLAMA_MODEL,
                         url: str = OLLAMA_URL,
                         timeout: float = TRANSLATE_TIMEOUT) -> Optional[str]:
    """Roman Urdu -> English via local Ollama. None on any failure.

    Raw urllib rather than the ollama SDK: one fewer dependency, and the
    SDK is not installed here. temperature 0 so the same complaint gives
    the same English every run - a translation step that drifts would make
    the downstream numbers irreproducible.

    Refusals are handled in three layers, because instructions alone did not
    hold: the record wrapper and the few-shot turns prevent most of them, one
    retry recovers the rest, and sanitize_translation() is the net that stops
    any survivor from being embedded as though it were a symptom.
    """
    text = (text or "").strip()
    if not text:
        return None
    # Dictionary pass first, so Ollama sees one spelling per word. Local,
    # deterministic, and logged.
    text = fuzzy_normalize_roman_urdu(text)

    # Refuse fragments BEFORE the model sees them. Asked to translate text
    # with no clinical content, it invents: "band ho rha ha" ("is closing")
    # came back as "Arm is swollen", "bhaag" ("run") as "Fainting". Cheaper
    # and more reliable to decline here than to catch the invention after -
    # and the anatomical gate can only catch the subset that names an organ.
    if not has_medical_signal(text):
        print(f"[guardrail] {text!r} contains no clinical word - refusing to "
              f"translate rather than inviting an invented symptom",
              flush=True)
        return None
    # Resolve to something actually installed. Calling a missing tag returns
    # a 404 that reads like a server fault rather than "you have no model".
    resolved = select_translation_model(preferred=model)
    if resolved is None:
        print("[ollama] no models installed - try: ollama pull llama3.2",
              flush=True)
        return None
    model = resolved

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for src, want in FEWSHOT_TURNS:
        messages.append({"role": "user", "content": USER_TEMPLATE.format(src)})
        messages.append({"role": "assistant", "content": want})
    messages.append({"role": "user", "content": USER_TEMPLATE.format(text)})

    for attempt in (1, 2):
        raw = _ollama_chat(messages, model, url, timeout)
        if raw is None:
            return None  # transport failure, not a refusal - do not retry
        cleaned = _strip_labels(raw)
        out = sanitize_translation(cleaned, text)
        if out is not None:
            return out
        if attempt == 1:
            print(f"[ollama] {model} refused; retrying once with an explicit "
                  f"non-refusal instruction", flush=True)
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": RETRY_NUDGE})
    print(f"[ollama] {model} refused twice - giving up on the translation. "
          f"Callers fall back to the untranslated text.", flush=True)
    return None


def _strip_labels(out: str) -> str:
    """Remove the label/quote wrappers small models add despite rule 5."""
    out = (out or "").strip()
    for prefix in ("English:", "Translation:", "English translation:",
                   "<record>", "Output:"):
        if out.lower().startswith(prefix.lower()):
            out = out[len(prefix):].strip()
    for suffix in ("</record>",):
        if out.lower().endswith(suffix.lower()):
            out = out[:-len(suffix)].strip()
    return out.strip().strip('"').strip()


#: Sentences that are the model talking ABOUT the task instead of doing it.
#:
#: Every pattern is deliberately anchored to a refusal-specific object -
#: "cannot translate", never bare "cannot". A patient complaint legitimately
#: translates to "I cannot walk" or "I am unable to breathe properly", and a
#: loose pattern would silently delete the most urgent line in the record.
#: These err towards KEEPING text: a stray disclaimer that slips through is
#: visible in the CLI output, whereas a deleted symptom is not.
_REFUSAL_PATTERNS = re.compile(
    r"""(?ix)
    (?:
        i \s* (?:'m|\s+am) \s+ (?:unable|not \s+ able) \s+ to \s+
            (?:translate|provide|assist|help|comply|process|do)
      | i \s+ (?:cannot|can't|can \s+ not) \s+
            (?:translate|provide|assist|help|comply|fulfill|process|do \s+ that)
      | i \s* (?:'m|\s+am) \s+ not \s+ (?:a \s+)?
            (?:medical|healthcare|licensed|qualified|doctor|physician|professional)
      | as \s+ an? \s+ (?:ai|language \s+ model|assistant)
      | i \s* (?:'m|\s+am) \s+ (?:sorry|afraid) \b
      | i \s+ apolog(?:ize|ise)
      | (?:can|how \s+ can|is \s+ there \s+ anything \s+ else) \s+ i \s+ help
      | please \s+ (?:consult|see \s+ a|contact|seek|call)
      | seek \s+ (?:immediate \s+ )? (?:medical|professional|emergency)
      | this \s+ is \s+ not \s+ (?:medical \s+ advice|a \s+ diagnosis)
      | \b disclaimer \b
      | i \s+ (?:do \s+ not|don't) \s+ have \s+ the \s+ ability
    )""")

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")

#: Below this, whatever survived the strip is too thin to be a translation
#: of a real complaint - treat it as nothing rather than embed a fragment.
MIN_TRANSLATION_CHARS = 8


def sanitize_translation(out: str, original: str = "") -> Optional[str]:
    """Drop refusals and disclaimers; return None if nothing real is left.

    Returning None matters more than it looks. Every caller already has a
    fallback for a failed translation - run() keeps the Roman Urdu
    prediction, preprocess_and_embed() normalises and embeds the ORIGINAL
    text - so None routes a refusal into the path that was built for it.
    Passing the refusal through instead would embed "I am not a medical
    professional" as though it were the patient's symptoms, and the triage
    score would then be describing the disclaimer.

    Sentence-level, not whole-output: models commonly emit a correct
    translation and then bolt "please consult a doctor" onto the end, and
    that case should keep the translation rather than discard the record.
    """
    out = (out or "").strip()
    if not out:
        return None

    parts = [p.strip() for p in _SENTENCE_SPLIT.split(out) if p.strip()]
    kept = [p for p in parts if not _REFUSAL_PATTERNS.search(p)]
    dropped = len(parts) - len(kept)

    cleaned = " ".join(kept).strip().strip('"').strip()
    if not cleaned:
        print(f"[guardrail] nothing left after stripping the model's reply "
              f"to {(original or '')[:60]!r}: {out[:120]!r}", flush=True)
        return None
    if dropped:
        print(f"[guardrail] dropped {dropped} disclaimer/refusal "
              f"sentence(s) from the model output", flush=True)

    # The length floor exists to catch LEFTOVERS - the fragment that
    # survives after a refusal is stripped out. It must not judge a reply
    # that arrived clean: "Fever" is a complete and correct translation of
    # "bukhar" at five characters, and this rejected it, so a real one-word
    # complaint produced no prediction at all. Only apply the floor when
    # something was actually removed.
    if dropped and len(cleaned) < MIN_TRANSLATION_CHARS:
        # The whole reply was a refusal. Say so loudly - a silent None here
        # looks identical to Ollama being down, and the two need different
        # fixes (prompt vs service).
        # flush=True on purpose: the GUI's failure dialog says "the console
        # shows the reason", and Python block-buffers stdout when it is a
        # pipe, so without this the console showed nothing at all.
        print(f"[guardrail] model REFUSED to translate "
              f"{(original or '')[:60]!r} - falling back to the untranslated "
              f"text. Full reply: {out[:160]!r}", flush=True)
        return None
    return cleaned


# ----------------------------------------------------------------------
# 1b. Dictionary normalization + the anatomical assertion gate
# ----------------------------------------------------------------------

#: Spelling variants collapsed onto one canonical Roman Urdu token before
#: the text ever reaches Ollama. The English glosses are documentation -
#: nothing translates from this table, the LLM still does that. Mapping to
#: a canonical ROMAN URDU key rather than to English is deliberate: it
#: leaves one spelling for the model to have seen, without pre-empting the
#: translation and without the train/serve skew that pushing English back
#: through a Roman Urdu dictionary caused before.
ROMAN_URDU_DICTIONARY = {
    # anatomy
    "pait": "stomach", "paet": "stomach", "pet": "stomach", "shikm": "stomach",
    "maida": "stomach", "mayda": "stomach", "payt": "stomach", "peit": "stomach",
    "seena": "chest", "seene": "chest", "chati": "chest", "chaati": "chest",
    "sar": "head", "sir": "head", "sear": "head", "khopri": "head",
    "gala": "throat", "galao": "throat", "gardan": "neck",
    "kamar": "back", "peeth": "back", "pith": "back",
    "kandha": "shoulder", "kandhay": "shoulder", "kandhe": "shoulder",
    "baazu": "arm", "bazu": "arm", "haath": "hand",
    "taang": "leg", "tang": "leg", "paon": "foot", "pair": "foot",
    "dil": "heart", "aankh": "eye", "kaan": "ear",
    # symptoms and modifiers
    "dard": "pain", "darad": "pain", "daard": "pain", "dukhna": "pain",
    "dardd": "pain", "dhard": "pain",
    "ghabrahat": "anxiety", "bechaini": "restlessness",
    "bukhar": "fever", "bookhar": "fever",
    "ulti": "vomiting", "oolti": "vomiting", "matli": "nausea",
    "pasina": "sweating", "paseena": "sweating",
    "chakkar": "dizziness", "behoshi": "fainting",
    "khansi": "cough", "sujan": "swelling", "chot": "injury",
    "shadeed": "severe",
}

#: Words that must NEVER be fuzzy-matched, measured against this project's
#: own 10,000-row corpus rather than guessed. Every entry here was observed
#: being remapped onto a medical term:
#:
#:     sa      -> sar      729x   comparison particle, "aag sa jalna"
#:     si      -> sir      452x   the same particle
#:     bhar    -> bukhar   191x   "din bhar", all day  ->  FEVER
#:     peene   -> seene    116x   "to drink"           ->  CHEST
#:     chalti  -> chati     24x   "walking"            ->  CHEST
#:     chadhti -> chati     24x   "climbing"           ->  CHEST
#:
#: This is the same failure that fuzzy matching produced here once before,
#: when it invented symptoms in 33.7% of rows. The lesson then and now: no
#: single cutoff separates a typo from a different word. Raising the cutoff
#: to 0.85 kills five of the six, and 'chalti' -> 'chati' survives even at
#: 0.90, because they genuinely are one edit apart. A threshold cannot fix
#: that; only naming the word can.
FUZZY_BLOCKLIST = {
    "sa", "si", "bhar", "peene", "peena", "chalti", "chadhti", "chadte",
    "chalte", "chal", "hai", "hay", "par", "pas", "kar", "kia", "kya",
    "raha", "rahi", "meri", "mera", "mein", "may", "aur", "phir", "tak",
    "koi", "kuch", "bohat", "buhat", "bahut", "zyada", "thora", "thoda",
}

#: A typo is a small perturbation of a longish word. Short tokens are
#: mostly function words, where one edit is the difference between two
#: unrelated words rather than a misspelling.
FUZZY_MIN_LEN = 4
FUZZY_CUTOFF = 0.88

#: Roman Urdu body part -> the English terms any faithful translation of it
#: must contain. Order does not matter; one match is enough.
ANATOMICAL_ASSERTIONS = {
    # Each key matches the body part in the SOURCE; the value lists the
    # English terms a faithful translation of it must contain. One match is
    # enough.
    #
    # THE SOURCE PATTERNS INCLUDE THE ENGLISH WORDS TOO, and that is not
    # redundancy. The complaint box accepts English - "chest mein pain ho
    # raha hai" is normal mixed usage, and the built-in examples are English
    # - but these patterns only listed Roman Urdu. So an English complaint
    # named no body part as far as the gate could see, its correct
    # translation named "chest", and the invented-anatomy rule fired on a
    # perfect translation. "chest mein pain ho raha hai" and "crushing chest
    # pain radiating to jaw" were both refused.
    r"\b(pait|paet|pet|payt|peit|shikm|maida|mayda|stomach|abdomen|abdominal|belly|tummy)\b":
        ["stomach", "abdomen", "abdominal", "belly"],
    r"\b(seena|seene|chati|chaati|chest|thorax|thoracic)\b":
        ["chest", "thorax", "thoracic"],
    r"\b(sar|sir|sear|khopri|head|skull|scalp)\b":
        ["head", "skull", "cranial", "scalp"],
    r"\b(gala|galao|gardan|throat|neck)\b":
        ["throat", "neck", "pharyn"],
    r"\b(peeth|pith|kamar|back|lumbar|spine)\b":
        ["back", "lumbar", "spine"],
    r"\b(kandha|kandhay|kandhe|shoulder)\b":
        ["shoulder"],
    r"\b(taang|tang|paon|pair|leg|foot|feet|ankle)\b":
        ["leg", "foot", "feet", "limb", "ankle"],
    r"\b(baazu|bazu|haath|hath|arm|hand|forearm|wrist)\b":
        ["arm", "hand", "forearm", "wrist"],
    r"\b(jabra|jabray|jaw)\b":
        ["jaw", "mandible"],
    # KNEE, and the trap that comes with it. "ghutna" is the knee; "ghutan"
    # is suffocation, and "dam ghutna" / "saans ghutna" mean choking, not
    # anything about a leg. The negative lookbehind keeps those phrases out,
    # so only a bare "ghutna" is read as the joint.
    #
    # Added because llama3.2 rendered "ghutna mein dard" (knee pain) as
    # "Difficulty breathing" and the gate had no knee entry to object with -
    # a knee complaint would have been triaged as respiratory.
    r"(?<!dam )(?<!dum )(?<!saans )\b(ghutna|ghutne|knee)\b":
        ["knee", "patella"],
}

#: Where the clinical vocabulary lives. A DATA file, deliberately, not a
#: Python literal: a maintainer adding "chakkar" for a new dialect should
#: edit JSON and save, not open a module and risk a syntax error in the
#: middle of a safety check.
CLINICAL_VOCAB_FILE = os.path.join(_ROOT, "clinical_vocabulary.json")

#: Loaded once and cached. The file is small and read at import, but the
#: cache matters for the batch path, where has_medical_signal() is called
#: once per row and a 500-row sheet would otherwise re-read and re-parse the
#: same JSON five hundred times.
_VOCAB_CACHE = None


def load_clinical_vocabulary(path: str = None) -> set:
    """Every word that marks a text as a complaint, from the JSON data file.

    Returns a lowercase set. Groups in the file are organisational only -
    they exist so a maintainer can see WHERE a new word belongs rather than
    appending to one flat list; nothing in the code depends on the grouping.

    Falls back to the dictionary keys if the file is missing or malformed
    rather than raising. A missing vocabulary file must not take the whole
    application down: the degraded behaviour is more refusals, which is the
    safe direction, and the console says what happened.
    """
    global _VOCAB_CACHE
    if _VOCAB_CACHE is not None and path is None:
        return _VOCAB_CACHE
    target = path or CLINICAL_VOCAB_FILE
    words = set()
    try:
        with open(target, "r", encoding="utf-8") as f:
            doc = json.load(f)
        for group in (doc.get("vocabulary") or {}).values():
            words.update(w.lower() for w in group if isinstance(w, str))
    except Exception as e:
        # Say so loudly. Silently falling back would make the system quietly
        # stricter than it was built to be, and nobody would know why real
        # complaints started being refused.
        print(f"[vocab] could not read {target} ({type(e).__name__}: {e}) - "
              f"falling back to the dictionary keys alone. Expect more "
              f"refusals until this is fixed.", flush=True)
    words |= {w.lower() for w in ROMAN_URDU_DICTIONARY}
    if path is None:
        _VOCAB_CACHE = words
    return words


def has_medical_signal(text: str) -> bool:
    """True when the text contains at least one clinically meaningful word.

    A record containing none is not a complaint - it is a fragment - and
    asking a language model to translate a fragment invites invention. Two
    measured examples from this project: "band ho rha ha" ("is closing")
    came back as "Arm is swollen"; "bhaag" ("run") came back as "Fainting".
    Neither had anything to translate.

    THE FAILURE DIRECTION IS CHOSEN. A word missing from the vocabulary
    causes a refusal, never a wrong triage level - the complaint goes to a
    clinician to read instead. For a triage system that is the correct way
    round: an unscored patient is an inconvenience, an invented symptom is a
    hazard.

    Punctuation is stripped per word because operators type "dard," and
    "bukhar." with trailing marks, and a set membership test would miss both.
    """
    vocab = load_clinical_vocabulary()
    words = {w.strip(".,;:!?()[]'\"").lower() for w in str(text or "").split()}
    return bool(words & vocab)


#: variant -> canonical Roman Urdu token, derived from the table above by
#: grouping on the English gloss and keeping the first spelling as canonical.
#:
#: Without this the dictionary was decorative. fuzzy_normalize_roman_urdu()
#: passed any word it recognised straight through unchanged, so "payt" and
#: "chaati" stayed exactly as typed and the only thing the table ever did
#: was serve as a list of fuzzy targets. Its docstring claimed variants were
#: "collapsed onto one canonical Roman Urdu token"; they were not, and the
#: anatomical gate was carrying the whole burden through its own regex
#: alternations.
_CANONICAL = {}
for _variant, _gloss in ROMAN_URDU_DICTIONARY.items():
    _CANONICAL[_variant] = next(k for k, v in ROMAN_URDU_DICTIONARY.items()
                                if v == _gloss)

_ANATOMICAL_COMPILED = [(re.compile(p, re.I), kws)
                        for p, kws in ANATOMICAL_ASSERTIONS.items()]
_WORD_SPLIT = re.compile(r"(\W+)")


def fuzzy_normalize_roman_urdu(text: str, verbose: bool = True) -> str:
    """Collapse spelling variants onto canonical Roman Urdu, typos included.

    Runs entirely locally, before Ollama sees anything. Punctuation and
    spacing are preserved - the string is split on word boundaries rather
    than whitespace, so "dard, ulti" does not become "dard ulti".

    Fuzzy matching here is deliberately timid. See FUZZY_BLOCKLIST for what
    an aggressive cutoff did to this corpus.
    """
    if not text:
        return text
    keys = list(ROMAN_URDU_DICTIONARY.keys())
    out, changes = [], []

    for token in _WORD_SPLIT.split(str(text)):
        low = token.lower()
        if not token.isalpha():
            out.append(token)
            continue
        if low in _CANONICAL:
            canon = _CANONICAL[low]
            out.append(canon)
            if canon != low:
                changes.append(f"{low}->{canon}")
            continue
        if low in FUZZY_BLOCKLIST or len(low) < FUZZY_MIN_LEN:
            out.append(token)
            continue
        match = get_close_matches(low, keys, n=1, cutoff=FUZZY_CUTOFF)
        # A typo should not change the word's length much. "chalti"/"chati"
        # slipped through on ratio alone; requiring a near-equal length as
        # well as a high ratio is a second, independent constraint.
        if match and abs(len(match[0]) - len(low)) <= 1:
            out.append(match[0])
            changes.append(f"{low}->{match[0]}")
        else:
            out.append(token)

    if changes and verbose:
        # Every substitution is announced. A normalizer that silently
        # rewrites a complaint is indistinguishable from one that corrupts
        # it, and this project has already shipped the corrupting version.
        print(f"[dictionary] fuzzy: {', '.join(changes)}", flush=True)
    return "".join(out)


def verify_anatomical_integrity(roman_urdu: str, translated_english: str) -> tuple:
    """Deterministic gate: every body part named in the source must survive.

    Returns (ok, list_of_failures). This replaces the cosine check as the
    DECISION. Cosine is still measured and reported, but it cannot decide
    anything: against "seena mein shadeed dard aur pasina", a correct
    translation scored 0.8054 and "My leg is broken after a fall" scored
    0.7922 - 0.013 apart, with the badly drifted candidate outscoring the
    mildly drifted one. No threshold separates those. This does, exactly,
    for the one failure mode that actually matters here.

    Pass the NORMALIZED Roman Urdu, not the raw string. A typo the
    dictionary just repaired ("payt" -> "pait") will not match these
    patterns in its raw form, and the gate would score a silent pass on
    precisely the input that most needed checking.

    Deliberately one-directional. It asserts that a named body part is
    present in the output; it does not assert that nothing else was added.
    An English translation may legitimately name a region the Roman Urdu
    only implied, and failing those would make the gate untrustworthy in
    the direction that gets it switched off.
    """
    if not translated_english:
        return False, ["no translation to check"]
    english_lower = translated_english.lower()
    failures = []
    source_named_any = False
    for pattern, keywords in _ANATOMICAL_COMPILED:
        found = pattern.search(roman_urdu or "")
        if found:
            source_named_any = True
            if not any(kw in english_lower for kw in keywords):
                failures.append(f"{found.group(0)!r} -> expected one of "
                                f"{keywords}, got none")

    # INVENTED ANATOMY. The check above only compares body parts the SOURCE
    # names, so a source naming none could not fail it - and that is exactly
    # where the worst output appeared: "band ho rha ha" ("is closing", no
    # body part, not really a complaint) came back as "Arm is swollen", and
    # the gate passed it because there was nothing to compare.
    #
    # Deliberately narrow. It fires ONLY when the source names no body part
    # at all and the English names one, which is unambiguous invention. When
    # the source does name a part, English is free to add related regions -
    # "seena mein dard" legitimately becoming "chest pain radiating to the
    # arm" - and that case is not touched. Widening this to "any part in
    # English not in the source" would reject good translations, which is
    # how a safety check ends up switched off.
    if not source_named_any:
        for pattern, keywords in _ANATOMICAL_COMPILED:
            hit = next((kw for kw in keywords if kw in english_lower), None)
            if hit:
                failures.append(
                    f"the complaint names no body part, but the translation "
                    f"says {hit!r} - invented anatomy")
                break

    return (not failures), failures


# ----------------------------------------------------------------------
# 2. Embeddings + classification
# ----------------------------------------------------------------------

class OfflinePredictor:
    """Encoder + the two RandomForest heads, loaded once."""

    def __init__(self, model_dir: str = MODEL_DIR):
        import joblib

        mpath = os.path.join(model_dir, "manifest.json")
        if not os.path.exists(mpath):
            raise SystemExit(
                f"No manifest at {mpath}. Train first:\n"
                f"    .venv/bin/python -m src.train")
        with open(mpath, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)

        # The encoder comes from the manifest, never from a constant here.
        enc = self.manifest["encoder"]
        self.encoder_name = enc["model"]
        self.prefix = enc.get("prefix", "")
        self.dim = enc["dim"]

        self.models = {}
        for target, meta in self.manifest["targets"].items():
            path = os.path.join(model_dir, meta["file"])
            if not os.path.exists(path):
                raise SystemExit(f"Manifest lists {meta['file']}, which is missing.")
            self.models[target] = joblib.load(path)
        self._st = None

    @property
    def st_model(self):
        if self._st is None:
            from sentence_transformers import SentenceTransformer
            try:
                self._st = SentenceTransformer(self.encoder_name, device="cpu",
                                               local_files_only=True)
            except Exception:
                # Only reachable on a machine that has never cached it.
                self._st = SentenceTransformer(self.encoder_name, device="cpu")
        return self._st

    def get_embedding(self, text: str) -> np.ndarray:
        """One L2-normalised sentence vector, matching training exactly."""
        prepared = self.prefix + (str(text).strip() or "empty complaint")
        vec = self.st_model.encode([prepared], convert_to_numpy=True,
                                   normalize_embeddings=True,
                                   show_progress_bar=False)[0]
        if vec.shape != (self.dim,):
            raise ValueError(f"encoder returned {vec.shape}, expected ({self.dim},)")
        return vec.astype(np.float32)

    def predict_triage_and_department(self, embedding: np.ndarray) -> dict:
        """Both heads from one vector. Confidence is the top class probability."""
        X = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        out = {}
        for target, model in self.models.items():
            pred = model.predict(X)[0]
            conf = None
            if hasattr(model, "predict_proba"):
                conf = float(np.max(model.predict_proba(X)[0]))
            out[target] = {"prediction": pred, "confidence": conf}
        return out



# ----------------------------------------------------------------------
# Offline text-to-speech
# ----------------------------------------------------------------------

#: eSpeak NG's default is ~175 words/min. 0.8x of that is 140, which is
#: the point of this feature: a triage nurse in a noisy room needs the
#: complaint read back deliberately, not at conversational pace.
TTS_BASE_WPM = 175
TTS_RATE = 0.8


def tts_available() -> bool:
    import shutil
    return shutil.which("espeak-ng") is not None or shutil.which("espeak") is not None


def speak(text: str, rate: float = TTS_RATE, blocking: bool = False) -> tuple:
    """Read `text` aloud through eSpeak NG. Returns (ok, message).

    Local binary, no network and no Python TTS dependency - the same
    offline constraint the rest of the pipeline lives under.

    Never raises: this is wired to a GUI button, and a missing audio
    device must not take the window down with it.
    """
    import shutil
    import subprocess

    text = (text or "").strip()
    if not text:
        return False, "nothing to speak"
    exe = shutil.which("espeak-ng") or shutil.which("espeak")
    if exe is None:
        return False, ("espeak-ng is not installed. Install it with:\n"
                       "    sudo dnf install espeak-ng")
    wpm = max(80, min(450, int(TTS_BASE_WPM * rate)))
    cmd = [exe, "-s", str(wpm), "--", text]
    try:
        if blocking:
            r = subprocess.run(cmd, capture_output=True, timeout=120)
            if r.returncode != 0:
                return False, r.stderr.decode("utf-8", "replace")[:200] or "espeak failed"
        else:
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
        return True, f"speaking at {wpm} wpm ({rate:g}x)"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


# ----------------------------------------------------------------------
# 3. Embedding alignment
# ----------------------------------------------------------------------

def cosine_similarity(vec1, vec2) -> float:
    a = np.asarray(vec1, dtype=np.float64).ravel()
    b = np.asarray(vec2, dtype=np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def verify_embedding_match(vec1, vec2, threshold: float = DEFAULT_THRESHOLD) -> bool:
    """True when two vectors are at least `threshold` cosine apart.

    Used to ask whether the translation still means what the original did.
    Note that 0.90 is demanding for a CROSS-LANGUAGE pair: the same
    complaint in Roman Urdu and English measures around 0.84 on this
    encoder even when the translation is perfect, so a False here is not
    by itself evidence of a bad translation - read the number, not the
    boolean.
    """
    return cosine_similarity(vec1, vec2) >= threshold


# ----------------------------------------------------------------------
# 4. End to end
# ----------------------------------------------------------------------

def run(text: str, reference: Optional[str] = None,
        model: str = OLLAMA_MODEL, model_dir: str = MODEL_DIR,
        threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Translate, embed, classify both languages, and compare.

    Returns a dict; the CLI renders it. Both predictions are included on
    purpose - see the module docstring for why the English one carries a
    caveat rather than standing alone.
    """
    predictor = OfflinePredictor(model_dir)
    # The tag actually used, not the one asked for - the fallback chain may
    # have chosen a different model, and a report that named the request
    # would be describing a translation that never happened.
    resolved = select_translation_model(preferred=model) or model
    result = {
        "input": text,
        "encoder": predictor.encoder_name,
        "ollama_model": resolved,
        "ollama_model_requested": model,
        "translation": None,
        "roman_urdu_prediction": None,
        "english_prediction": None,
        "similarity": {},
        "notes": [],
    }

    vec_ru = predictor.get_embedding(text)
    result["roman_urdu_prediction"] = predictor.predict_triage_and_department(vec_ru)

    english = translate_roman_urdu(text, model=resolved)
    result["translation"] = english
    if english is None:
        result["notes"].append(
            "Translation failed - the Roman Urdu prediction above still "
            "stands, since it is the one that matches how the classifiers "
            "were trained.")
        return result

    vec_en = predictor.get_embedding(english)
    result["english_prediction"] = predictor.predict_triage_and_department(vec_en)

    sim = cosine_similarity(vec_ru, vec_en)
    result["similarity"]["roman_urdu_vs_english"] = sim
    result["similarity"]["passes_threshold"] = sim >= threshold
    result["similarity"]["threshold"] = threshold
    result["similarity"]["is_gate"] = False

    # THE GATE. Deterministic anatomical assertion, on the NORMALIZED source:
    # a typo the dictionary just repaired would not match the patterns in its
    # raw form, and the gate would pass exactly the input that needed it most.
    normalized_source = fuzzy_normalize_roman_urdu(text, verbose=False)
    ok, failures = verify_anatomical_integrity(normalized_source, english)
    result["anatomical_gate"] = {"passed": ok, "failures": failures,
                                 "checked_source": normalized_source}

    # Cosine is measured and reported, and decides nothing. It was the gate
    # until it was measured: a correct translation of "seena mein shadeed
    # dard aur pasina" scores 0.8054 and "My leg is broken after a fall"
    # scores 0.7922, so no threshold separates them. Keeping the number
    # visible is useful; keeping it in charge was not.
    result["accepted_source"] = "english" if ok else "roman_urdu"
    result["accepted_prediction"] = (result["english_prediction"] if ok
                                     else result["roman_urdu_prediction"])
    if ok:
        result["notes"].append(
            "ANATOMICAL GATE PASSED: every body part named in the complaint "
            "survives into the English, so the ENGLISH prediction is the "
            "accepted one. The gate checks anatomy, not overall fidelity - "
            "it cannot see a wrong symptom attached to the right body part.")
    else:
        result["notes"].append(
            "ANATOMICAL GATE FAILED: " + "; ".join(failures) + ". The "
            "translation moved the complaint to a different part of the body, "
            "so the ROMAN URDU prediction is the accepted one.")

    if reference:
        vec_ref = predictor.get_embedding(reference)
        result["reference"] = reference
        result["similarity"]["english_vs_reference"] = cosine_similarity(vec_en, vec_ref)
        result["similarity"]["roman_urdu_vs_reference"] = cosine_similarity(vec_ru, vec_ref)

    ru = result["roman_urdu_prediction"]
    en = result["english_prediction"]
    won = "English" if result["accepted_source"] == "english" else "Roman Urdu"
    for target in ru:
        if ru[target]["prediction"] != en[target]["prediction"]:
            result["notes"].append(
                f"{target}: Roman Urdu says {ru[target]['prediction']}, English "
                f"says {en[target]['prediction']}. The gate accepted the "
                f"{won} one. Read both - a disagreement here is the two heads "
                f"splitting on the same patient, not a resolved question.")
    return result
