import requests
import os

# ============================================================
# SETTINGS
# ============================================================

API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not API_KEY:
    raise SystemExit("Set OPENROUTER_API_KEY before running this script.")

MODEL = "openai/gpt-4o-mini"

URL = "https://openrouter.ai/api/v1/chat/completions"


# ============================================================
# PROMPT
#
# Rules 1-6 are the original six, unchanged. Rules 7-10 were added after
# reading all 50 translations from the first sample run, and each one
# exists to close a defect that run actually produced - they are not
# speculative tightening:
#
#   7  "Samajh nahi aa raha" is the writer hedging ("can't quite tell"),
#      but it came back as "I am feeling confused" in 4 of 50 rows (8%).
#      Confusion is a red-flag neurological finding. The model was
#      inventing an acuity-relevant symptom that is not in the source,
#      which is rule 2 being broken in the direction that matters most.
#      Row 15 dropped the phrase correctly, so the behaviour was
#      inconsistent rather than uniform.
#
#   8  "kasav" (constriction) and "khichao" (tightness) were flattened to
#      generic "chest pain" in rows 36 and 50. Tightness versus pain is
#      exactly the distinction the canonical dictionary keeps separate
#      (jákṛan / dabāo / dárd), so losing it in translation would hand the
#      English arm of this comparison a handicap the Roman Urdu arm does
#      not have.
#
#   9  "aaram se kam ho jata hai" (resolves WITH REST) became "gradually
#      improving" in row 11 - a syncope complaint, where how it resolves
#      changes how urgent it reads.
#
#  10  "tez chalne par" (on walking fast) became "worsens with walking" in
#      row 44. The source states a trigger, not a worsening.
#
# ORIGINAL_PROMPT_TEMPLATE is kept verbatim so the two can be compared,
# and so the diff of this experiment is auditable.
# ============================================================

ORIGINAL_PROMPT_TEMPLATE = """
Translate the following Roman Urdu medical chief complaint into
natural English.

Rules:
1. Preserve the exact clinical meaning.
2. Do not add information that is not present.
3. Do not make a diagnosis.
4. Preserve symptoms, duration and severity.
5. Return ONLY the English translation.
6. Do not provide explanations.

Roman Urdu complaint:
"""

PROMPT_TEMPLATE = """
Translate the following Roman Urdu medical chief complaint into
natural English.

Rules:
1. Preserve the exact clinical meaning.
2. Do not add information that is not present.
3. Do not make a diagnosis.
4. Preserve symptoms, duration and severity.
5. Return ONLY the English translation.
6. Do not provide explanations.
7. The specific opening phrase "samajh nahi aa raha" expresses the
   writer's uncertainty. Translate that phrase alone as "unclear" or omit
   it, and never as patient confusion, disorientation or altered mental
   state - that is a symptom the source does not report. This applies ONLY
   to that phrase. Every other clause containing "nahi" is an ordinary
   negation and must be translated literally: "neend nahi aa rahi" is
   "unable to sleep", "bhook nahi lag rahi" is "no appetite", "bukhar
   nahi" is "no fever". Do not carry uncertainty wording into them.
8. Preserve the exact quality of the sensation. "kasav", "jakran" and
   "khichao" are tightness or constriction, "dabao" is pressure, "bhaari
   pan" is heaviness, "chubhan" is a pricking or stabbing sensation and
   "jalan" is burning. Do not replace any of these with generic "pain".
9. "aaram se kam ho jata hai" and "rest se theek ho jata hai" mean the
   symptom improves WITH REST. Do not translate them as improving
   gradually, on its own, or over time.
10. Preserve trigger relationships exactly as stated. "X par" or "X karte
   waqt" means the symptom occurs ON or DURING X. Do not upgrade this to
   "worsens with X" unless the source says it worsens.

Roman Urdu complaint:
"""


# ============================================================
# TRANSLATION FUNCTION
# ============================================================

def translate_roman_urdu(text):

    headers = {
        "Authorization": "Bearer " + API_KEY,
        "Content-Type": "application/json"
    }

    prompt = PROMPT_TEMPLATE + text

    data = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0,
        "max_tokens": 200
    }

    try:
        response = requests.post(
            URL,
            headers=headers,
            json=data,
            timeout=60
        )
    except Exception as e:
        print("\nConnection error:")
        print(e)
        return None

    if response.status_code != 200:
        print("\nAPI Error:")
        print("Status code:", response.status_code)
        print(response.text)
        return None

    try:
        result = response.json()
    except Exception as e:
        print("\nCould not read API response:")
        print(e)
        print(response.text)
        return None

    try:
        choices = result.get("choices")

        if not choices:
            print("\nNo choices returned by the API.")
            print(result)
            return None

        message = choices[0].get("message")

        if not message:
            print("\nNo message returned by the API.")
            print(result)
            return None

        translation = message.get("content")

        if translation is None:
            print("\nAPI returned no translation.")
            print("Full response:")
            print(result)
            return None

        return translation.strip()

    except Exception as e:
        print("\nError extracting translation:")
        print(e)
        print("Full response:")
        print(result)
        return None


# ============================================================
# BATCH TRANSLATION  (step 2 - wraps translate_roman_urdu above)
#
# The function above is one complaint at a time. This one walks a CSV
# column and reuses it unchanged, so both paths send byte-identical
# prompts and the experiment measures the prompt, not two variants of it.
#
# Saves incrementally. A run over 10,000 rows is long enough that a
# dropped connection or a rate limit partway through is likely, and
# losing an hour of completed (and paid-for) translations to a crash is
# the failure mode worth engineering against. Rows already translated in
# an existing output file are skipped on a re-run, so the same command
# resumes rather than starting over.
# ============================================================

TRANSLATION_COLUMN = "English_Translation"
SAVE_EVERY = 10


def translate_dataset_column(csv_path, text_column, output_csv,
                             sample_size=None, random_state=42,
                             max_workers=1):
    """Translate `text_column` of `csv_path` into `output_csv`.

    Args:
        csv_path: source CSV.
        text_column: column holding the Roman Urdu text.
        output_csv: destination. Written incrementally, and re-read on
            startup so an interrupted run resumes where it stopped.
        sample_size: translate a random sample of this many rows instead
            of the whole file. None means every row.
        random_state: seed for that sample, so the same 50 rows come back
            on a re-run.

    Returns:
        The DataFrame that was written.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    if text_column not in df.columns:
        raise SystemExit(
            f"Column {text_column!r} not in {csv_path}. "
            f"Available: {list(df.columns)}")

    if sample_size is not None and sample_size < len(df):
        df = df.sample(sample_size, random_state=random_state).reset_index(drop=True)

    # Resume: keep any translation already present in the output file.
    done = {}
    if os.path.exists(output_csv):
        try:
            prev = pd.read_csv(output_csv)
            if TRANSLATION_COLUMN in prev.columns:
                done = {row[text_column]: row[TRANSLATION_COLUMN]
                        for _, row in prev.iterrows()
                        if isinstance(row.get(TRANSLATION_COLUMN), str)
                        and row[TRANSLATION_COLUMN].strip()}
                print(f"[resume] {len(done)} translations already in {output_csv}")
        except Exception as e:
            print(f"[resume] could not read {output_csv}, starting fresh: {e}")

    if TRANSLATION_COLUMN not in df.columns:
        df[TRANSLATION_COLUMN] = None

    total = len(df)
    translated = failed = skipped = 0

    # Rows already done are filled straight in; only the rest hit the API.
    pending = []
    for i, row in df.iterrows():
        source = str(row[text_column])
        if source in done:
            df.at[i, TRANSLATION_COLUMN] = done[source]
            skipped += 1
        else:
            pending.append((i, source))

    if max_workers > 1 and pending:
        # Concurrency exists for one reason: 10,000 rows at ~4 s each is
        # roughly 12 hours serial, which is not a run anyone will babysit.
        # Requests are independent and the bottleneck is round-trip latency,
        # so threads are the right tool. Saves still happen on the main
        # thread as results land, keeping the resume guarantee intact.
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading
        lock = threading.Lock()
        print(f"  translating {len(pending)} rows with {max_workers} workers")
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(translate_roman_urdu, src): (i, src)
                       for i, src in pending}
            for n, fut in enumerate(as_completed(futures), 1):
                i, _src = futures[fut]
                try:
                    result = fut.result()
                except Exception as e:
                    print(f"\nWorker error on row {i}: {e}")
                    result = None
                with lock:
                    if result is None:
                        failed += 1
                    else:
                        df.at[i, TRANSLATION_COLUMN] = result
                        translated += 1
                    if n % SAVE_EVERY == 0 or n == len(pending):
                        df.to_csv(output_csv, index=False)
                        print(f"  {n}/{len(pending)} translated "
                              f"(ok {translated}, failed {failed})"
                              f"  -> saved to {output_csv}")
    else:
        for n, (i, source) in enumerate(pending, 1):
            result = translate_roman_urdu(source)
            if result is None:
                # translate_roman_urdu has already printed the reason.
                failed += 1
            else:
                df.at[i, TRANSLATION_COLUMN] = result
                translated += 1
            if n % SAVE_EVERY == 0 or n == len(pending):
                df.to_csv(output_csv, index=False)
                print(f"  {n}/{len(pending)} rows  "
                      f"(translated {translated}, resumed {skipped}, failed {failed})"
                      f"  -> saved to {output_csv}")

    df.to_csv(output_csv, index=False)
    print(f"\n[done] {total} rows: {translated} translated, {skipped} resumed, "
          f"{failed} failed. Written to {output_csv}")
    if failed:
        print(f"[warn] {failed} row(s) have an empty {TRANSLATION_COLUMN}. "
              f"Re-run the same command to retry only those.")
    return df


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":
    print()
    print("=" * 60)
    print("Roman Urdu → English Medical Complaint Translator")
    print("=" * 60)
    print()
    print("Type 'exit' or 'quit' to stop.")
    print()

    while True:
        complaint = input("Roman Urdu complaint: ").strip()

        if complaint.lower() in ["exit", "quit"]:
            print("Exiting...")
            break

        if complaint == "":
            continue

        print()
        print("Translating...")

        translation = translate_roman_urdu(complaint)

        if translation is not None:
            print()
            print("English:", translation)

        print()
