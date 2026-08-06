# Domain-Guided Lightweight Feature Attention
# Hybrid Text Representation for Roman Urdu Emergency Triage Classification
# (BoW + Fuzzy Matching + Diacritization Version)
#
# The text pipeline (diacritization map, fuzzy vocab, attention weights,
# normalization) now lives in triage_pipeline.py so that training and every
# predictor share ONE identical definition and can never drift apart.

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer   # BoW replaces TF-IDF
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import json
import os
import warnings
import matplotlib.pyplot as plt
from collections import Counter

# Shared text pipeline (single source of truth)
from triage_pipeline import (
    normalize_roman_urdu,
    build_attention_weights,
    make_console_safe,
    NUMERICAL_FEATURES,
)

warnings.filterwarnings('ignore')

# This script prints the diacritized vocabulary ('bukhār', 'dárd', 'sēna'),
# which the default Windows console codepage cannot encode. Without this the
# run dies with UnicodeEncodeError partway through the word-frequency table,
# before the model is ever saved.
make_console_safe()

os.makedirs('triage_model', exist_ok=True)
os.makedirs('visualizations', exist_ok=True)

print("Loading dataset...")
df = pd.read_csv('triage_mixed_language_dataset.csv')
print(f"  {len(df)} patient records loaded.")

# ============================================================
# TEXT NORMALIZATION
# normalization -> fuzzy matching -> diacritization
# (all defined once in triage_pipeline.py)
# ============================================================

print("Normalizing complaints (fuzzy + diacritization, multi-category vocab)...")
df['Complaint_Text'] = df['Complaint_Text'].fillna('').apply(normalize_roman_urdu)

# ============================================================
# WORD FREQUENCY HISTOGRAM (ASCENDING ORDER)
# ============================================================

print("\n" + "=" * 70)
print("GENERATING WORD FREQUENCY HISTOGRAM")
print("=" * 70)

all_complaints = ' '.join(df['Complaint_Text'].astype(str))
words = all_complaints.split()

stop_words = {'hai', 'hain', 'tha', 'hey', 'g', 'ka', 'ki', 'ke', 'ko', 'se', 'mein',
              'ne', 'to', 'bhi', 'nahi', 'na', 'ho', 'tha', 'thi', 'the', 'raha',
              'rahi', 'rahe', 'kar', 'kya', 'wo', 'ye'}

filtered_words = [w for w in words if w not in stop_words and len(w) > 1]
word_freq = Counter(filtered_words)

top_n = 30
top_words = word_freq.most_common()[:top_n]
top_words_ascending = list(reversed(top_words))

words_list  = [item[0] for item in top_words_ascending]
frequencies = [item[1] for item in top_words_ascending]

plt.figure(figsize=(14, 8))
bars = plt.barh(words_list, frequencies, color='steelblue', edgecolor='black', alpha=0.7)
plt.xlabel('Frequency of Occurrence', fontsize=12, fontweight='bold')
plt.ylabel('Words', fontsize=12, fontweight='bold')
plt.title(f'Top {top_n} Most Frequent Words in Chief Complaints\n(Ascending Order - Least to Most Frequent)',
          fontsize=14, fontweight='bold', pad=20)

for bar, freq in zip(bars, frequencies):
    plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
             str(freq), va='center', fontsize=9)

plt.grid(axis='x', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('visualizations/word_frequency_histogram.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"\nWord Frequency Statistics:")
print(f"Total unique words: {len(word_freq)}")
print(f"Total word occurrences: {sum(word_freq.values())}")
print(f"\nTop {top_n} words (ascending order):")
for i, (word, freq) in enumerate(top_words_ascending, 1):
    print(f"  {i:2d}. '{word}' - {freq} occurrences")

word_freq_df = pd.DataFrame(word_freq.most_common(), columns=['Word', 'Frequency'])
word_freq_df.to_csv('visualizations/word_frequencies.csv', index=False)
print(f"\n[ok] Word frequency data saved to 'visualizations/word_frequencies.csv'")
print(f"[ok] Histogram saved to 'visualizations/word_frequency_histogram.png'")

# ============================================================
# CATEGORICAL ENCODING
# ============================================================

le_gender = LabelEncoder()
le_mode   = LabelEncoder()
le_avpu   = LabelEncoder()
le_ecg    = LabelEncoder()

df['Gender_enc'] = le_gender.fit_transform(df['Gender'])
df['Mode_enc']   = le_mode.fit_transform(df['Mode_of_Arrival'])
df['AVPU_enc']   = le_avpu.fit_transform(df['AVPU'])
df['ECG_enc']    = le_ecg.fit_transform(df['ECG_Status'])

# ============================================================
# NUMERICAL FEATURES
# ============================================================

# Assign the filled column back explicitly. `df[col].fillna(..., inplace=True)`
# is chained assignment on a temporary Series: pandas 2.x only warns (and that
# warning is swallowed by filterwarnings('ignore') above), but in pandas 3.0 it
# is a silent no-op, so any NaN vital sign would survive into StandardScaler and
# poison every feature row with NaN.
for col in NUMERICAL_FEATURES:
    df[col] = df[col].fillna(df[col].median())

scaler = StandardScaler()
df[NUMERICAL_FEATURES] = scaler.fit_transform(df[NUMERICAL_FEATURES])

# ============================================================
# BAG OF WORDS (BoW) TEXT REPRESENTATION
# Word-level BoW  -> captures medical terminology
# Char-level BoW  -> captures sub-word patterns (handles misspellings)
# ============================================================

complaints = df['Complaint_Text'].astype(str)

word_bow = CountVectorizer(max_features=350, ngram_range=(1, 2), analyzer='word')
char_bow = CountVectorizer(max_features=350, ngram_range=(2, 4), analyzer='char_wb')

word_features = word_bow.fit_transform(complaints).toarray()
char_features = char_bow.fit_transform(complaints).toarray()
text_features = np.hstack([word_features, char_features])

# ============================================================
# DOMAIN-GUIDED LIGHTWEIGHT FEATURE ATTENTION
# Weights come from triage_pipeline.MEDICAL_WEIGHTS (shared).
# ============================================================

feature_names = (list(word_bow.get_feature_names_out()) +
                 list(char_bow.get_feature_names_out()))
weights = build_attention_weights(feature_names)
text_features = text_features * weights

# ============================================================
# COMBINE FEATURES
# ============================================================

structured_features = df[
    NUMERICAL_FEATURES + ['Gender_enc', 'Mode_enc', 'AVPU_enc', 'ECG_enc']
].values

X = np.hstack([structured_features, text_features])
y = df['Triage_Level'].values - 1   # dataset 1..4  ->  model classes 0..3

# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================================
# MODEL
# ============================================================

model = LogisticRegression(max_iter=1200, class_weight='balanced')
model.fit(X_train, y_train)
pred = model.predict(X_test)

# ============================================================
# CONFUSION MATRIX AND TRIAGE QUALITY METRICS
# ============================================================

print("\n" + "=" * 70)
print("TRIAGE CLASSIFICATION RESULTS")
print("=" * 70)

accuracy = accuracy_score(y_test, pred)
print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

cm = confusion_matrix(y_test, pred, labels=[0, 1, 2, 3])
print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)
print("(Rows = True Labels, Columns = Predicted Labels)")
print(f"{'':>12} {'Pred 0':>8} {'Pred 1':>8} {'Pred 2':>8} {'Pred 3':>8}")
print("-" * 50)
for i in range(len(cm)):
    print(f"True {i:<6} {cm[i, 0]:>8} {cm[i, 1]:>8} {cm[i, 2]:>8} {cm[i, 3]:>8}")

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)
print(classification_report(
    y_test, pred, labels=[0, 1, 2, 3],
    target_names=['Level 0 (Emergency)', 'Level 1 (Urgent)',
                  'Level 2 (Standard)', 'Level 3 (Non-Urgent)']
))

# ============================================================
# OVER-TRIAGE AND UNDER-TRIAGE CALCULATIONS
# ============================================================

print("\n" + "=" * 70)
print("TRIAGE QUALITY METRICS: OVER-TRIAGE & UNDER-TRIAGE ANALYSIS")
print("=" * 70)

triage_metrics = {}

for true_class in range(4):
    tp_count     = cm[true_class, true_class]
    total_actual = np.sum(cm[true_class, :])

    over_triage_count  = sum(cm[true_class, p] for p in range(true_class))
    under_triage_count = sum(cm[true_class, p] for p in range(true_class + 1, 4))

    over_triage_rate  = (over_triage_count  / total_actual) * 100 if total_actual > 0 else 0
    under_triage_rate = (under_triage_count / total_actual) * 100 if total_actual > 0 else 0
    correct_rate      = (tp_count           / total_actual) * 100 if total_actual > 0 else 0

    triage_metrics[true_class] = {
        'total_actual':      int(total_actual),
        'correct':           int(tp_count),
        'over_triage':       int(over_triage_count),
        'over_triage_rate':  float(over_triage_rate),
        'under_triage':      int(under_triage_count),
        'under_triage_rate': float(under_triage_rate),
        'correct_rate':      float(correct_rate)
    }

    class_names = {
        0: "Level 0 (Emergency - Most Urgent)",
        1: "Level 1 (Urgent)",
        2: "Level 2 (Standard)",
        3: "Level 3 (Non-Urgent - Least Urgent)"
    }

    print(f"\n{class_names[true_class]}:")
    print(f"  Total patients in test set: {total_actual}")
    print(f"  Correctly classified: {tp_count} ({correct_rate:.1f}%)")

    if true_class == 3:
        print(f"  [x] OVER-TRIAGE (more urgent than needed): {over_triage_count} ({over_triage_rate:.1f}%)")
        print(f"  [ok] UNDER-TRIAGE (less urgent): {under_triage_count} ({under_triage_rate:.1f}%)")
    else:
        print(f"  [!] OVER-TRIAGE: {over_triage_count} ({over_triage_rate:.1f}%)")
        print(f"  [!] UNDER-TRIAGE: {under_triage_count} ({under_triage_rate:.1f}%)")

    if true_class == 0:
        if under_triage_rate < 10:
            print(f"  [ok] SAFETY: Under-triage rate {under_triage_rate:.1f}% (<10%) - Good")
        else:
            print(f"  [x] SAFETY RISK: Under-triage rate {under_triage_rate:.1f}% (>10%) - Dangerous!")
    elif true_class == 3:
        if over_triage_rate < 30:
            print(f"  [!] RESOURCE IMPACT: Over-triage rate {over_triage_rate:.1f}% (<30%) - Acceptable")
        else:
            print(f"  [x] RESOURCE WASTE: Over-triage rate {over_triage_rate:.1f}% (>30%) - Too many false alarms")

# ============================================================
# OVERALL TRIAGE QUALITY SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("OVERALL TRIAGE QUALITY SUMMARY")
print("=" * 70)

total_patients     = len(y_test)
over_triage_total  = sum(triage_metrics[i]['over_triage']  for i in range(4))
under_triage_total = sum(triage_metrics[i]['under_triage'] for i in range(4))
correct_total      = sum(triage_metrics[i]['correct']      for i in range(4))

over_triage_overall  = (over_triage_total  / total_patients) * 100
under_triage_overall = (under_triage_total / total_patients) * 100
correct_overall      = (correct_total      / total_patients) * 100

print(f"\nTotal test patients: {total_patients}")
print(f"Correctly triaged: {correct_total} ({correct_overall:.1f}%)")
print(f"Over-triaged: {over_triage_total} ({over_triage_overall:.1f}%)")
print(f"Under-triaged (SAFETY RISK): {under_triage_total} ({under_triage_overall:.1f}%)")

print("\n" + "-" * 50)
print("SAFETY GRADE:")
if   under_triage_overall < 5:  print("   A+ (Excellent) - Under-triage rate <5%")
elif under_triage_overall < 10: print("   A (Good) - Under-triage rate <10%")
elif under_triage_overall < 15: print("   B (Acceptable) - Under-triage rate <15%")
elif under_triage_overall < 20: print("   C (Caution) - Under-triage rate <20%")
else:                           print("   F (Unsafe) - Under-triage rate >20% - DO NOT DEPLOY")

print("\nRESOURCE EFFICIENCY GRADE:")
if   over_triage_overall < 15: print("   A (Excellent) - Over-triage rate <15%")
elif over_triage_overall < 25: print("   B (Good) - Over-triage rate <25%")
elif over_triage_overall < 35: print("   C (Acceptable) - Over-triage rate <35%")
else:                          print("   D (Poor) - Over-triage rate >35%")

# ============================================================
# DETAILED MISCLASSIFICATION PATTERNS
# ============================================================

print("\n" + "=" * 70)
print("DETAILED MISCLASSIFICATION PATTERNS")
print("=" * 70)

class_names_short = ['L0 (Emerg)', 'L1 (Urgent)', 'L2 (Std)', 'L3 (NonU)']

for true_class in range(4):
    print(f"\nTrue {class_names_short[true_class]}:")
    total = np.sum(cm[true_class, :])
    for pred_class in range(4):
        if true_class != pred_class and cm[true_class, pred_class] > 0:
            count      = cm[true_class, pred_class]
            percentage = (count / total) * 100
            direction  = "UPGRADED (over-triage)" if pred_class < true_class else "DOWNGRADED (under-triage)"
            print(f"  -> {count} patients ({percentage:.1f}%) {direction} to {class_names_short[pred_class]}")

# ============================================================
# SAVE MODEL AND METRICS
# ============================================================

print("\n" + "=" * 70)
print("SAVING MODEL AND METRICS")
print("=" * 70)

joblib.dump(model,    'triage_model/model.pkl')
joblib.dump(word_bow, 'triage_model/word_bow.pkl')
joblib.dump(char_bow, 'triage_model/char_bow.pkl')
joblib.dump(scaler,   'triage_model/scaler.pkl')
joblib.dump(le_gender, 'triage_model/gender_enc.pkl')
joblib.dump(le_mode,   'triage_model/mode_enc.pkl')
joblib.dump(le_avpu,   'triage_model/avpu_enc.pkl')
joblib.dump(le_ecg,    'triage_model/ecg_enc.pkl')

metrics_summary = {
    'accuracy':                    float(accuracy),
    'overall_over_triage_rate':    float(over_triage_overall),
    'overall_under_triage_rate':   float(under_triage_overall),
    'correct_classification_rate': float(correct_overall),
    'total_patients':              int(total_patients),
    'confusion_matrix':            cm.tolist(),
    'per_class_metrics': {
        f'class_{i}': {
            'triage_level':      i,
            'total_actual':      triage_metrics[i]['total_actual'],
            'over_triage_rate':  triage_metrics[i]['over_triage_rate'],
            'under_triage_rate': triage_metrics[i]['under_triage_rate'],
            'correct_rate':      triage_metrics[i]['correct_rate']
        } for i in range(4)
    }
}

with open('triage_model/triage_metrics.json', 'w') as f:
    json.dump(metrics_summary, f, indent=4)

print("[ok] Model saved to 'triage_model/'")
print("[ok] BoW vectorizers saved (word_bow.pkl, char_bow.pkl)")
print("[ok] Label encoders saved")
print("[ok] Triage metrics saved as 'triage_metrics.json'")

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)
print("\nVisualization Summary:")
print("   - Word frequency histogram -> 'visualizations/word_frequency_histogram.png'")
print("   - Word frequency data      -> 'visualizations/word_frequencies.csv'")
