# Features — What, How, and Why

A plain-English guide to everything this project does. No maths background
needed. Each feature has three short parts:

1. **What it does** — the plain description
2. **How it works** — the simple version
3. **Why we added it** — the problem it solves

> **The one-line summary.** A patient walks into an emergency department in
> Pakistan and describes their problem in Roman Urdu ("seena mein dard" —
> chest pain). This program reads that sentence, combines it with their vital
> signs, and says how urgently they need to be seen. It runs on an ordinary
> laptop with no internet.

---

## Table of Contents

1. [Triage levels — what the answer means](#1-triage-levels--what-the-answer-means)
2. [Batch prediction from a file](#2-batch-prediction-from-a-file)
3. [Dictionary + fuzzy spelling matching](#3-dictionary--fuzzy-spelling-matching)
4. [Automatic stop-word removal (Contribution 1)](#4-automatic-stop-word-removal-contribution-1)
5. [Embeddings — letting AI read the sentence](#5-embeddings--letting-ai-read-the-sentence)
6. [The hybrid model](#6-the-hybrid-model)
7. [Embedding-evaluation study (Contribution 2)](#7-embedding-evaluation-study-contribution-2)
8. [The desktop app (GUI)](#8-the-desktop-app-gui)
9. [Two safety ideas used everywhere](#9-two-safety-ideas-used-everywhere)

---

## 1. Triage levels — what the answer means

The program sorts patients into four levels.

| Level | Name | Meaning |
|---|---|---|
| 1 | EMERGENCY | Needs attention right now. Life-threatening. |
| 2 | URGENT | Should be seen within 15 minutes. |
| 3 | STANDARD | Should be seen within 60 minutes. |
| 4 | NON-URGENT | Can wait, or be sent elsewhere. |

Two kinds of mistake matter, and they are **not** equally bad:

- **Under-triage** — the program says a patient is *less* urgent than they
  really are. A very sick person gets sent to the back of the queue. **This is
  the dangerous mistake.**
- **Over-triage** — the program says a patient is *more* urgent than they
  really are. Nobody is harmed, but staff time is wasted.

Everything in this project treats under-triage as the more serious error.

---

## 2. Batch prediction from a file

**What it does.** Triages many patients at once — 100, 500, or more — from a
single Excel or CSV file, instead of typing them in one at a time.

**How it works.** You put one patient per row in a spreadsheet. The only column
that must be filled in is `Complaint_Text`. The program reads every row, runs
each patient through the same steps, and writes a new file with extra columns
added: the predicted level, the label, how confident it is, and the probability
of each of the four levels. If a number is missing it quietly fills in the
average from the training data, and if a category value was never seen before
it falls back to a safe known value. Either way you still get an answer, and
the substitution is written into that row's `Notes` column so you can check it.

**Why we added it.** A real emergency department does not have time to type
patients in one by one. It also means the whole day's patients can be reviewed
afterwards in Excel. Filling in missing values instead of crashing matters
because real hospital data is always messy — one blank cell should not stop the
other 99 patients from being triaged.

**Try it:** `python predict_batch.py sample_100_patients.xlsx`, or use the
**Batch File** tab in the app.

---

## 3. Dictionary + fuzzy spelling matching

**What it does.** Understands the same word even when it is spelled many
different ways.

**How it works.** Roman Urdu has no official spelling. *Fever* can be written
`bukhar`, `bukhaar`, `bukhr`, `bkhar`, or `garmi`. The program fixes this in
three passes:

1. **Cleaning** — makes everything lowercase and removes punctuation.
2. **Fuzzy matching** — compares each word against a list of known medical
   words. If a word is at least 80% similar to a known one, it is swapped for
   it. So `bukhr` becomes `bukhar`.
3. **Diacritization** — every known spelling of a word is collapsed onto one
   single canonical form. `dard`, `dardh`, `durd`, and `drd` all become `dárd`.

After this, five different spellings of "pain" have become one word that the
model can actually count.

**Why we added it.** Without it, the model treats `dard` and `drd` as two
completely unrelated words, so it learns almost nothing from either. This step
is what makes Roman Urdu usable at all. It is kept even in the newest
embedding-based version, because the AI model was trained mostly on English and
proper Urdu script, so it does not handle Roman Urdu misspellings on its own.

**See it happen:** the **Pipeline Explorer** tab shows a sentence moving through
every one of these stages.

---

## 4. Automatic stop-word removal (Contribution 1)

**What it does.** Finds and removes the filler words that carry no medical
meaning — and works out *which* words those are by itself, from the data.

**How it works.** A word is removed only if **both** of these are true:

1. **It is common** — it turns up in a lot of complaints.
2. **It tells you nothing about urgency** — knowing whether the word is present
   does not help predict the triage level. This is measured two ways (mutual
   information and a chi-square test), and both must agree.

Every threshold and every number behind every decision is saved in
`learned_stopwords.json`, so anyone can check the working by hand.

**Why we added it.** There is no published stop-word list for Roman Urdu, and
writing one by hand is guesswork — someone's opinion, not evidence. Learning it
from the data makes it reproducible and defensible in a research paper.

**Two results worth knowing:**

- The obvious filler words (`hai`, `mein`, `aur`) were **not** removed. On this
  dataset they genuinely do carry a signal, because how people phrase things
  turns out to relate to how sick they are. An honest method keeps them. A
  hand-written list would have thrown them away.
- The maths on its own wanted to delete real symptom words like `pain`,
  `jalan` (burning) and `khoon` (blood), because they appear at every triage
  level and so look "uninformative". Deleting a symptom from a triage system is
  the one mistake worth engineering against, so a **clinical safety guard**
  protects the medical vocabulary. Every word it rescues is listed in the file,
  so the guard is visible rather than hidden.

**See it:** the **Stop Words** tab shows the full table and why each word was
kept or dropped.

---

## 5. Embeddings — letting AI read the sentence

**What it does.** Turns a whole complaint into a list of numbers that captures
its *meaning*, so that sentences meaning the same thing end up close together.

**How it works.** A small offline AI model reads the sentence and outputs 384
numbers. Think of it as giving every sentence a position on a map. Sentences
about chest pain should land near other chest-pain sentences, and far from
sentences about a broken ankle. Nothing is looked up in a dictionary — the model
works it out from the sentence itself.

The model is downloaded once and then runs entirely on your laptop's CPU, with
no internet and no graphics card.

**Why we added it.** A hand-written dictionary can only ever recognise the
spellings someone remembered to add. A patient who writes something new is
invisible to it. Embeddings are meant to handle words the dictionary never saw.

**Honest note.** These AI models are trained mostly on English and proper Urdu
script (اردو). *Roman* Urdu is under-represented, so this works less well here
than it would in English. That is exactly what Feature 7 measures, rather than
assuming.

**See it:** the **Embedding Demo** on the **Model Score** tab turns your own
sentence into numbers in front of you.

---

## 6. The hybrid model

**What it does.** Uses the dictionary **and** the embeddings together, instead
of choosing one.

**How it works.** The dictionary features and the 384 embedding numbers are
joined into one long list, along with the vital signs (age, heart rate, blood
pressure, temperature, oxygen, consciousness, ECG). That whole list goes into
the classifier, which outputs the triage level.

**Why we added it.** The two methods fail in different ways. The dictionary is
reliable on the spellings it knows and useless on the ones it does not.
Embeddings are the opposite — broad, but vaguer on Roman Urdu. Keeping both
means the dictionary still anchors the words the AI model misses.

**Which one is actually used?** The program trains all four options on the same
patients and the same split, then picks a winner using a safety-first rule:
**lowest under-triage first, accuracy only as the tie-breaker.** The live
numbers are in `embedding_pipeline_results.csv` and are shown on the
**Model Score** tab — this document deliberately does not repeat them, so it
can never go stale or disagree with the real files.

---

## 7. Embedding-evaluation study (Contribution 2)

**What it does.** Measures how good the embedding model actually is at
understanding Roman Urdu complaints. It does not change the triage prediction at
all — it is a measurement.

**How it works.** Three checks:

1. **Do similar complaints land close together?** Ten groups of complaints were
   sorted by meaning by hand (chest pain, fever, burns, and so on). Inside each
   group, every complaint is compared with every other complaint, and we count
   how many pairs score above 0.5 similarity.

2. **Do different complaints stay apart?** High similarity inside a group proves
   nothing on its own — a broken model that called *everything* similar would
   score perfectly. So the average similarity *between* different groups is also
   measured. The gap between the two is the real evidence.

3. **The round-trip test.** For each complaint, find its single closest
   neighbour out of all the others, then check whether that neighbour actually
   means the same thing. The percentage that get it right is the
   **embedding generator efficiency** — the headline number for the paper.

**Why the round-trip test is done this way.** The original idea was "turn the
text into numbers, then turn the numbers back into text and check it matches".
That is not possible: this kind of embedding is one-way and lossy — the numbers
simply do not contain enough to rebuild the sentence. The nearest-neighbour
version measures the same underlying thing (does the vector faithfully represent
the meaning?) and can actually be run.

**Why we added it.** A reviewer said the project's contribution was too thin.
This turns "we used AI embeddings" into a measured claim with numbers behind it.
It also honestly exposes where the model is weak on Roman Urdu, which points at
the next step (translating to Urdu script, or fine-tuning on our own data).

**See it:** the **Results** tab and the **Model Score** tab both read the live
numbers from `embedding_evaluation_results.csv`.

---

## 8. The desktop app (GUI)

**What it does.** Puts everything behind buttons, and — more importantly —
*shows the working*, not just the answer.

**How it works.** Six tabs:

| Tab | What it gives you |
|---|---|
| **Triage a Patient** | Type one patient, get the level and confidence, plus what the text pipeline did to their complaint |
| **Pipeline Explorer** | Type any sentence and watch every cleaning stage happen, with the statistics behind each removed word |
| **Stop Words** | The learned list as a table — removed, kept for signal, or rescued by the safety guard |
| **Batch File** | Triage a whole spreadsheet, with a results table and level counts |
| **Results** | The four-method comparison and the per-cluster embedding scores as charts |
| **Model Score & Embedding Analysis** | The real accuracy figures, a visual explanation of the "45 pairs" maths, and a live embedding demo |

It is built with `tkinter`, which comes free with Python, so it needs **no extra
installation** and stays fully offline.

**Why we added it.** Two reasons. First, a doctor or a teammate should not need
to use a terminal. Second, and more useful for the research: the tabs make the
method *visible*. It is far easier to trust — or to challenge — a step you can
watch happen to your own sentence than a paragraph describing it.

Every number in the app is read from the result files on disk. If a file has not
been generated yet, the app says which command to run. **It never shows an
invented figure.**

**Run it:** `python triage_gui.py`

---

## 9. Two safety ideas used everywhere

These come up in several features, so they are worth stating once.

**Under-triage outranks accuracy.** When two options disagree — one more
accurate, one safer — the safer one wins. A model that is half a percent more
accurate but sends more critically ill people to the back of the queue is not
the better model for an emergency department.

**Never invent a number.** Every figure shown in the app or written in a results
file comes from an actual run on actual data. Where something has not been
measured yet, the program says so instead of filling the gap. Where a result is
disappointing — and some of the Roman Urdu embedding results are — it is
reported as it is, not tuned until it looks better.

---

> **Reminder:** this is a research and educational prototype, not a certified
> medical device. It must never be the only basis for a clinical decision.
> Always involve a qualified clinician.
