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

**How it works.** A word is removed only if **all three** of these are true:

1. **It is common** — it turns up in a lot of complaints.
2. **Its mutual information with the triage level is near zero** — knowing the
   word is present barely narrows down how urgent the patient is.
3. **A chi-square test cannot tell it apart from independent** — there is no
   statistical evidence the word relates to the triage level at all.

Points 2 and 3 measure similar things but do not always agree, and **both must
pass**. That is not a formality: it is exactly what saves `hai` (see below).

This means **"stop words removed" does not mean "all filler removed"**. On this
dataset the learned list is ten words — `baad, bhi, jaisa, ki, lekin, nahi,
saath, se, tak, tez` — and plenty of ordinary filler survives, on purpose.

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

  Worth knowing *which* test saved them, because it differs. `aur` fails both
  tests — it plainly tracks urgency. `hai` is subtler: its mutual information
  is 0.0062, *below* the 0.01 cut-off, so by that measure it looks like filler.
  What keeps it is the chi-square test, p = 0.0007 — far too small to call it
  independent of the triage level. The **Stop Words** tab names the deciding
  test on every row rather than lumping all of them together as "carries
  signal", which would contradict the mutual-information column sitting right
  next to it.
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
was meant to let the dictionary anchor the words the AI model misses.

**Does it work?** Not on this dataset. Once the two feature blocks were
rescaled so the classifier could actually see both (they were not, for a while —
see below), the hybrid scored *worse* than either method on its own. It is kept
because measuring it is the point, not because it won.

**Which one is actually used?** **Method C, embeddings + preprocessing.** The
program trains all four options on the same patients and the same split and
reports a safety-first recommendation, but what ships is an explicit choice:
the `--deploy` flag, default `C`. The app names the deployed method on the
Triage, Batch, Results and Model Score tabs, and every command-line predictor
prints it at startup. The live numbers are in `embedding_pipeline_results.csv`
and on the **Model Score** tab — this document deliberately does not repeat
them, so it can never go stale or disagree with the real files.

**Why deployment is a separate decision.** The safety rule alone used to decide
this, and it could not: when two methods tie it silently keeps the first, which
is the dictionary baseline. Combined with the scaling fault below, that made the
"embedding pipeline" ship a dictionary-only model — reporting itself as the
embedding pipeline while recording `embedding_model: null` in its own metrics.
Separating "what scores best" from "what ships" is what makes that failure
impossible to repeat quietly.

> **The scaling fault, recorded rather than hidden.** The dictionary features
> are multiplied by domain attention weights; the embedding features are
> L2-normalised across 384 dimensions. Joined into one list without rescaling,
> the two blocks sit on very different scales, and a single penalised classifier
> responds by ignoring the smaller one entirely — the hybrid reproduced the
> dictionary-only result to the last decimal. Each block is now standardised
> separately before being joined, and the training script prints both blocks'
> scales on every run so the fault cannot creep back unnoticed.

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
| **Triage a Patient** | Type one patient, get the level and confidence, plus what the text pipeline did to their complaint and which stage the live model was fed |
| **Pipeline Explorer** | Type any sentence and watch every cleaning stage happen. Each stage is tagged *changed this text* or *ran — nothing to change here*, so a stage that had nothing to correct is never mistaken for a stage that did not run |
| **Stop Words** | Every tested token as a table, with the specific criterion that decided it — removed, kept because chi-square says it tracks triage, kept because its mutual information is too high, or rescued by the clinical safety guard |
| **Batch File** | Triage a whole spreadsheet, with a results table and level counts |
| **Results** | The four-method comparison and the per-cluster embedding scores as charts, each row labelled with the pipeline that produced it |
| **Model Score & Embedding Analysis** | The real accuracy figures for both pipelines side by side, a visual explanation of the "45 pairs" maths, and a live embedding demo |

**Which model is live is stated, never implied.** A green banner on the Triage,
Batch, Results and Model Score tabs names the deployed method, the text features
it uses and the directory it came from. Score cards are marked **LIVE** or
**not deployed**, and each carries a coloured "Numbers produced by:" line, so a
dictionary figure can never be read as an embedding figure.

It is built with `tkinter`, which ships with Python on Windows and macOS, so it
needs **no extra installation** and stays fully offline. Some Linux
distributions package it separately — `sudo dnf install python3-tkinter` on
Fedora, `sudo apt install python3-tk` on Debian/Ubuntu.

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
