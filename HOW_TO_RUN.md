# How To Run This Program — Simple Step-by-Step Guide

This guide is written for someone who has **never used Python or a terminal
before**. Follow the steps in order. You only do Steps 1–3 once on a computer.
After that, you just repeat Step 4 whenever you want to triage patients.

If you get stuck, jump to the **"If something goes wrong"** section at the bottom.

---

## What this program does (in one line)

You give it a file full of patients (their complaint + vitals), and it tells you
how urgent each patient is (Level 1 = Emergency ... Level 4 = Non-urgent),
all at once.

---

## Step 1 — Install Python (one time only)

The program is written in Python, so the computer needs Python installed.

1. Go to **https://www.python.org/downloads/**
2. Click the big yellow **"Download Python"** button.
3. Open the file you downloaded to start the installer.
4. **VERY IMPORTANT:** On the first screen, tick the box at the bottom that says
   **"Add Python to PATH"** before clicking Install. If you miss this box,
   the commands later won't work.
5. Click **Install Now** and wait for it to finish, then close the installer.

To check it worked: open the **Start menu**, type `powershell`, open
**Windows PowerShell**, type this and press Enter:

```
python --version
```

If it prints something like `Python 3.12.0`, you are good. If it says
"not recognized", reinstall Python and make sure the **"Add Python to PATH"**
box is ticked.

---

## Step 2 — Put the project folder on the computer

1. You were given a file called **`ED_updated.zip`**.
2. Right-click it and choose **"Extract All..."**, then **Extract**.
3. This gives you a folder named `ED_updated`. Inside it is another folder also
   named `ED_updated` that contains all the program files
   (`predict_batch.py`, `README.md`, etc.). **That inner folder is the one you
   work in.**

Tip: move this folder somewhere easy to find, like your **Desktop** or
**Downloads**.

---

## Step 3 — Open a terminal *inside* the project folder (one time per session)

The program runs by typing a command, and the command must be typed **inside the
folder that has the program files**. The easiest way on Windows:

1. Open the `ED_updated` folder that contains `predict_batch.py` (the inner one).
2. Click once on the **address bar** at the top of the window (where it shows the
   folder path).
3. Type `powershell` over it and press **Enter**.
4. A blue/black window opens. It is already "inside" the correct folder. 

Now install the few things the program needs (this downloads them once):

```
pip install -r requirements.txt
```

Wait for it to finish. You'll see some text scroll by — that's normal. You only
need to do this once per computer.

---

## Step 4 — Run the triage (this is the part you repeat)

Still in that same PowerShell window, type:

```
python predict_batch.py sample_100_patients.xlsx
```

Press Enter. In a couple of seconds you will see a summary like this:

```
[ok] 100 patient rows found.
==============================================================================
BATCH TRIAGE SUMMARY
==============================================================================
  Level 1 (EMERGENCY ) :   38  ######################################
  Level 2 (URGENT    ) :   30  ##############################
  Level 3 (STANDARD  ) :   28  ############################
  Level 4 (NON-URGENT) :    4  ####
  TOTAL                  :  100
```

**That means it worked — it triaged 100 patients at once.**

The full results were saved into the same folder as two files:

- `sample_100_patients_predictions.xlsx`  ← open this one in Excel
- `sample_100_patients_predictions.csv`

Open the `.xlsx` file. Each patient now has new columns added:
**Predicted_Triage_Level**, **Predicted_Label**, **Confidence**, the four
probability columns, and a **Notes** column.

---

## Step 5 — Run it on YOUR OWN patients

The example file is just to prove it works. To triage your own list:

1. In the folder, find **`batch_input_template.xlsx`** and open it in Excel.
2. Click **File → Save As** and save it as **`my_patients.xlsx`** in the **same
   folder** (the one with `predict_batch.py`).
3. Keep the top header row exactly as it is. Delete the example rows under it and
   type your own patients — **one patient per row**.
   - The only column you MUST fill is **`Complaint_Text`** (the symptom text,
     e.g. `seena mein dard aur pasina`).
   - The other columns (Age, Heart_Rate, etc.) are optional but make it more
     accurate. You can leave a cell blank if you don't have it.
4. Save and close the file.
5. Back in PowerShell, run:

```
python predict_batch.py my_patients.xlsx
```

It creates **`my_patients_predictions.xlsx`** with the answers.

> The file name in the command must match your file name **exactly**, and the
> file must be in the same folder you opened PowerShell in.

---

## A 30-second test to prove it's really working

Make a small file with two obvious patients:

- A clear emergency: `seena dard aur paseena bohat aa raha hai` with a low SpO2
  (e.g. 88) and high Heart_Rate (e.g. 130).
- A clear minor case: `sore throat aur mild fever`.

Run the batch command on it. The emergency should come out **Level 1** and the
minor case **Level 3 or 4**. If it does, the model is behaving sensibly.

---

## If something goes wrong

**"python is not recognized" / "pip is not recognized"**
Python isn't installed correctly. Redo Step 1 and make sure the
**"Add Python to PATH"** box is ticked during install. Then close PowerShell and
open it again.

**`[error] Input file not found: my_patients.xlsx`**
The program can't find your file. Two usual reasons:
1. You haven't created `my_patients.xlsx` yet (it's just an example name) — do
   Step 5 first.
2. The file is in a different folder. It must sit next to `predict_batch.py`.
   In PowerShell, type `dir` and press Enter to list the files in the current
   folder — if you don't see your file in that list, it's in the wrong place.

**`ModuleNotFoundError: No module named 'openpyxl'` (or pandas, sklearn, etc.)**
You skipped the install step. Run `pip install -r requirements.txt` (Step 3).

**`InconsistentVersionWarning` or
`AttributeError: 'LogisticRegression' object has no attribute 'multi_class'`**
Your computer has a different version of scikit-learn than the saved model.
Easiest fix — type:
```
pip install scikit-learn==1.6.1
```
(Or, if you want to keep your newer version, run
`python triage_bow_fuzzy_diac.py` once to rebuild the model on your version.)

**It opened the wrong folder in PowerShell**
The command only works if PowerShell is "inside" the folder with
`predict_batch.py`. Close it and redo Step 3 (type `powershell` in the folder's
address bar).

---

## Quick command cheat-sheet

| I want to...                              | Type this                                          |
|-------------------------------------------|----------------------------------------------------|
| Install everything (once)                 | `pip install -r requirements.txt`                  |
| Triage the 100 example patients           | `python predict_batch.py sample_100_patients.xlsx` |
| Triage my own file                        | `python predict_batch.py my_patients.xlsx`         |
| Triage one patient by hand (Q&A style)    | `python prediction_interactive.py`                 |
| See files in the current folder           | `dir`                                              |
| Check Python is installed                 | `python --version`                                 |

---

> Reminder: this is a student / research project for learning, not a certified
> medical device. The "Confidence" number is the model's guess, not a medical
> guarantee — always have a real clinician make the final call.
