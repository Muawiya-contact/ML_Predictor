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
# TRANSLATION FUNCTION
# ============================================================

def translate_roman_urdu(text):

    headers = {
        "Authorization": "Bearer " + API_KEY,
        "Content-Type": "application/json"
    }

    prompt = """
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
""" + text

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
                             sample_size=None, random_state=42):
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

    for i, row in df.iterrows():
        source = str(row[text_column])

        if source in done:
            df.at[i, TRANSLATION_COLUMN] = done[source]
            skipped += 1
            continue

        result = translate_roman_urdu(source)

        if result is None:
            # translate_roman_urdu has already printed the reason.
            failed += 1
        else:
            df.at[i, TRANSLATION_COLUMN] = result
            translated += 1

        if (i + 1) % SAVE_EVERY == 0 or (i + 1) == total:
            df.to_csv(output_csv, index=False)
            print(f"  {i + 1}/{total} rows  "
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
