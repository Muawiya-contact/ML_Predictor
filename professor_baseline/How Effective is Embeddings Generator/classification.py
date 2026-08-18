import os  # added for the redacted API key
import requests
import warnings

import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    LabelEncoder
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

import matplotlib.pyplot as plt
import seaborn as sns


warnings.filterwarnings("ignore")


# ============================================================
# SETTINGS
# ============================================================

CSV_FILE = "dataset.csv"


# ============================================================
# OPENROUTER
# ============================================================

# Put your NEW OpenRouter API key here.
#
# IMPORTANT:
# The API key previously posted in this conversation should
# be revoked/rotated because it has been exposed.

API_KEY = os.environ.get("OPENROUTER_API_KEY", "")  # REDACTED: a live key was hardcoded here

MODEL = "openai/gpt-4o-mini"

URL = "https://openrouter.ai/api/v1/chat/completions"


# ============================================================
# SENTENCE TRANSFORMER
# ============================================================

EMBEDDING_MODEL = "intfloat/multilingual-e5-small"


# ============================================================
# MACHINE LEARNING SETTINGS
# ============================================================

TEST_SIZE = 0.20

RANDOM_STATE = 42

MIN_CLASS_COUNT = 2


# ============================================================
# OUTPUT FILES
# ============================================================

PROCESSED_DATA_FILE = (
    "fused_dataset.xlsx"
)

EMBEDDING_FILE = (
    "complaint_embeddings.npy"
)

TRIAGE_MODEL_RESULTS = (
    "triage_classifier_results.xlsx"
)

DEPARTMENT_MODEL_RESULTS = (
    "department_classifier_results.xlsx"
)

TRIAGE_CONFUSION_MATRIX = (
    "triage_confusion_matrix.png"
)

DEPARTMENT_CONFUSION_MATRIX = (
    "department_confusion_matrix.png"
)


# ============================================================
# CHECK API KEY
# ============================================================

if (
    API_KEY == ""
    or
    API_KEY == "PUT_YOUR_NEW_OPENROUTER_KEY_HERE"
):

    print()
    print("=" * 70)
    print("ERROR")
    print("=" * 70)
    print()
    print(
        "Please put your OpenRouter API key in API_KEY."
    )
    print()
    exit()


# ============================================================
# READ DATASET
# ============================================================

print()
print("=" * 70)
print("READING DATASET")
print("=" * 70)
print()


try:

    df = pd.read_csv(
        CSV_FILE
    )

except Exception as e:

    print(
        "Could not read dataset.csv"
    )

    print(e)

    exit()


print(
    "Total records:",
    len(df)
)

print(
    "Number of columns:",
    len(df.columns)
)

print()


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

if len(df.columns) < 13:

    print(
        "ERROR: dataset.csv must contain at least 13 columns."
    )

    print(
        "The experiment requires columns A through M."
    )

    exit()


# ============================================================
# COLUMN MAPPING
# ============================================================

# Python column numbering starts from 0.
#
# A = 0
# B = 1
# C = 2
# D = 3  --> Chief Complaint
# E = 4
# F = 5
# G = 6
# H = 7
# I = 8
# J = 9
# K = 10
# L = 11 --> Triage Level
# M = 12 --> Department


COMPLAINT_COLUMN = 3

TRIAGE_COLUMN = 11

DEPARTMENT_COLUMN = 12


# Structured input columns:
#
# A, B, C, E, F, G, H, I, J, K
#
# D is deliberately excluded because D is represented by
# the Sentence Transformer embedding.

STRUCTURED_COLUMNS = [
    0,
    1,
    2,
    4,
    5,
    6,
    7,
    8,
    9,
    10
]


# ============================================================
# DISPLAY COLUMN NAMES
# ============================================================

print("=" * 70)
print("COLUMN MAPPING")
print("=" * 70)
print()


print(
    "Chief Complaint :",
    df.columns[COMPLAINT_COLUMN]
)

print(
    "Triage Level    :",
    df.columns[TRIAGE_COLUMN]
)

print(
    "Department      :",
    df.columns[DEPARTMENT_COLUMN]
)

print()


print(
    "Structured input columns:"
)

for column_index in STRUCTURED_COLUMNS:

    print(
        f"  {column_index + 1}. "
        f"{df.columns[column_index]}"
    )

print()


# ============================================================
# EXTRACT BASIC DATA
# ============================================================

chief_complaints = (

    df.iloc[:, COMPLAINT_COLUMN]

    .fillna("")

    .astype(str)

    .str.strip()
)


triage_levels = (

    df.iloc[:, TRIAGE_COLUMN]

    .fillna("")

    .astype(str)

    .str.strip()
)


departments = (

    df.iloc[:, DEPARTMENT_COLUMN]

    .fillna("")

    .astype(str)

    .str.strip()
)


# ============================================================
# REMOVE RECORDS WITH MISSING CORE INFORMATION
# ============================================================

valid_mask = (

    (chief_complaints != "")

    &

    (triage_levels != "")

    &

    (departments != "")
)


df = df.loc[
    valid_mask
].reset_index(drop=True)


chief_complaints = (

    df.iloc[:, COMPLAINT_COLUMN]

    .fillna("")

    .astype(str)

    .str.strip()
)


triage_levels = (

    df.iloc[:, TRIAGE_COLUMN]

    .fillna("")

    .astype(str)

    .str.strip()
)


departments = (

    df.iloc[:, DEPARTMENT_COLUMN]

    .fillna("")

    .astype(str)

    .str.strip()
)


print(
    "Records retained:",
    len(df)
)

print()


# ============================================================
# TARGET DISTRIBUTIONS
# ============================================================

print("=" * 70)
print("TRIAGE LEVEL DISTRIBUTION")
print("=" * 70)
print()


print(
    triage_levels.value_counts().to_string()
)

print()


print("=" * 70)
print("DEPARTMENT DISTRIBUTION")
print("=" * 70)
print()


print(
    departments.value_counts().to_string()
)

print()


# ============================================================
# TRANSLATION FUNCTION
# ============================================================

def translate_roman_urdu(text):

    headers = {

        "Authorization":
            "Bearer " + API_KEY,

        "Content-Type":
            "application/json"
    }


    prompt = """
Translate the following Roman Urdu medical chief complaint into
natural English.

Rules:

1. Preserve the exact clinical meaning.
2. Do not add information that is not present.
3. Do not make a diagnosis.
4. Preserve symptoms, duration and severity.
5. Correct obvious Roman Urdu spelling variations when necessary.
6. Use natural medical English.
7. Return ONLY the English translation.
8. Do not provide explanations.

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

        "max_tokens": 100
    }


    # --------------------------------------------------------
    # SEND REQUEST
    # --------------------------------------------------------

    try:

        response = requests.post(

            URL,

            headers=headers,

            json=data,

            timeout=60
        )

    except Exception as e:

        print()
        print("Connection error:")
        print(e)

        return None


    # --------------------------------------------------------
    # CHECK RESPONSE
    # --------------------------------------------------------

    if response.status_code != 200:

        print()
        print("API ERROR")

        print(
            "Status code:",
            response.status_code
        )

        print(
            response.text
        )

        return None


    # --------------------------------------------------------
    # READ JSON
    # --------------------------------------------------------

    try:

        result = response.json()

    except Exception as e:

        print()
        print(
            "Could not read API response:"
        )

        print(e)

        return None


    # --------------------------------------------------------
    # EXTRACT TRANSLATION
    # --------------------------------------------------------

    try:

        choices = result.get(
            "choices"
        )


        if not choices:

            print(
                "No choices returned."
            )

            return None


        message = choices[0].get(
            "message"
        )


        if not message:

            print(
                "No message returned."
            )

            return None


        translation = message.get(
            "content"
        )


        if translation is None:

            print(
                "API returned no translation."
            )

            return None


        return translation.strip()


    except Exception as e:

        print()
        print(
            "Error extracting translation:"
        )

        print(e)

        return None


# ============================================================
# TRANSLATE EVERY COMPLAINT USING API
# ============================================================

print()
print("=" * 70)
print("TRANSLATING ROMAN URDU COMPLAINTS USING OPENROUTER API")
print("=" * 70)
print()


translations = []


print(
    "Total complaints to translate:",
    len(chief_complaints)
)

print()


# IMPORTANT:
#
# There is deliberately NO cache here.
#
# Every row is sent to the OpenRouter API.
#
# Even if the same complaint appears multiple times,
# every occurrence generates a new API request.

for number, complaint in enumerate(

    chief_complaints,

    start=1

):

    print(
        f"[{number}/{len(chief_complaints)}] "
        "TRANSLATING:"
    )

    print(
        "Roman Urdu:",
        complaint
    )


    translation = (

        translate_roman_urdu(

            complaint

        )

    )


    if translation is None:

        print(
            "WARNING: Translation failed."
        )

        translation = ""


    print(
        "English:",
        translation
    )

    print()


    translations.append(
        translation
    )


print()
print("=" * 70)
print("TRANSLATION COMPLETE")
print("=" * 70)
print()


print(
    "API translations performed:",
    len(translations)
)

print()


# ============================================================
# CREATE WORKING DATAFRAME
# ============================================================

results = df.copy()


results[
    "English_Translation"
] = translations


# ============================================================
# LOAD SENTENCE TRANSFORMER
# ============================================================

print()
print("=" * 70)
print("LOADING SENTENCE TRANSFORMER")
print("=" * 70)
print()


print(
    "Model:",
    EMBEDDING_MODEL
)

print()


try:

    embedding_model = SentenceTransformer(

        EMBEDDING_MODEL

    )

except Exception as e:

    print(
        "Could not load Sentence Transformer."
    )

    print(e)

    exit()


# ============================================================
# GENERATE COMPLAINT EMBEDDINGS
# ============================================================

print()
print("=" * 70)
print("GENERATING COMPLAINT EMBEDDINGS")
print("=" * 70)
print()


english_texts = (

    results[
        "English_Translation"
    ]

    .fillna("")

    .tolist()

)


# E5 models work better when the input is explicitly identified
# as a passage.

e5_texts = [

    "passage: " + text

    if text.strip() != ""

    else "passage: empty complaint"

    for text in english_texts

]


embeddings = embedding_model.encode(

    e5_texts,

    convert_to_numpy=True,

    normalize_embeddings=True,

    show_progress_bar=True,

    batch_size=32

)


print()


print(
    "Embedding shape:",
    embeddings.shape
)


# ============================================================
# SAVE EMBEDDINGS
# ============================================================

np.save(

    EMBEDDING_FILE,

    embeddings

)


print(
    "Embeddings saved:",
    EMBEDDING_FILE
)

print()


# ============================================================
# PREPARE STRUCTURED FEATURES
# ============================================================

print("=" * 70)
print("PREPARING STRUCTURED FEATURES")
print("=" * 70)
print()


structured_df = df.iloc[

    :,

    STRUCTURED_COLUMNS

].copy()


structured_column_names = (

    list(
        structured_df.columns
    )

)


print(
    "Structured features:"
)

for column in structured_column_names:

    print(
        " ",
        column
    )

print()


# ============================================================
# IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ============================================================

numeric_columns = (

    structured_df

    .select_dtypes(

        include=["number"]

    )

    .columns

    .tolist()

)


categorical_columns = [

    column

    for column in structured_df.columns

    if column not in numeric_columns

]


print(
    "Numerical columns:"
)

print(
    numeric_columns
)

print()


print(
    "Categorical columns:"
)

print(
    categorical_columns
)

print()


# ============================================================
# CONVERT CATEGORICAL DATA TO STRING
# ============================================================

for column in categorical_columns:

    structured_df[column] = (

        structured_df[column]

        .fillna("MISSING")

        .astype(str)

    )


# ============================================================
# STRUCTURED DATA PREPROCESSOR
# ============================================================

numeric_pipeline = Pipeline([

    (
        "imputer",

        SimpleImputer(

            strategy="median"

        )

    ),

    (
        "scaler",

        StandardScaler()

    )

])


categorical_pipeline = Pipeline([

    (
        "imputer",

        SimpleImputer(

            strategy="most_frequent"

        )

    ),

    (
        "onehot",

        OneHotEncoder(

            handle_unknown="ignore"

        )

    )

])


transformers = []


if numeric_columns:

    transformers.append(

        (

            "numeric",

            numeric_pipeline,

            numeric_columns

        )

    )


if categorical_columns:

    transformers.append(

        (

            "categorical",

            categorical_pipeline,

            categorical_columns

        )

    )


preprocessor = ColumnTransformer(

    transformers=transformers

)


# ============================================================
# FIT STRUCTURED PREPROCESSOR
# ============================================================

print(
    "Encoding structured features..."
)


structured_features = (

    preprocessor.fit_transform(

        structured_df

    )

)


print(
    "Structured feature matrix shape:",
    structured_features.shape
)

print()


# ============================================================
# CONVERT STRUCTURED MATRIX TO DENSE
# ============================================================

if hasattr(

    structured_features,

    "toarray"

):

    structured_features = (

        structured_features.toarray()

    )


# ============================================================
# FUSE EMBEDDINGS + STRUCTURED FEATURES
# ============================================================

print("=" * 70)
print("FUSING FEATURES")
print("=" * 70)
print()


fused_features = np.hstack([

    embeddings,

    structured_features

])


print(
    "Sentence embedding dimensions:",
    embeddings.shape[1]
)


print(
    "Structured feature dimensions:",
    structured_features.shape[1]
)


print(
    "TOTAL fused dimensions:",
    fused_features.shape[1]
)

print()


# ============================================================
# SAVE FUSED DATASET
# ============================================================

processed_results = results.copy()


processed_results[
    "Triage_Level"
] = triage_levels


processed_results[
    "Department"
] = departments


processed_results[
    "Embedding_Row"
] = range(

    len(
        processed_results
    )

)


processed_results.to_excel(

    PROCESSED_DATA_FILE,

    index=False

)


print(
    "Processed data saved:",
    PROCESSED_DATA_FILE
)

print()


# ============================================================
# CLASSIFICATION FUNCTION
# ============================================================

def train_classifier(

    X,

    y,

    target_name,

    confusion_matrix_file,

    excel_file

):

    print()
    print("=" * 70)

    print(
        f"TRAINING CLASSIFIER: {target_name}"
    )

    print("=" * 70)
    print()


    # --------------------------------------------------------
    # REMOVE CLASSES WITH TOO FEW INSTANCES
    # --------------------------------------------------------

    class_counts = (

        y.value_counts()

    )


    valid_classes = (

        class_counts[

            class_counts >= MIN_CLASS_COUNT

        ].index

    )


    valid_mask = y.isin(

        valid_classes

    )


    X_valid = X[

        valid_mask.values

    ]


    y_valid = y[

        valid_mask

    ].reset_index(

        drop=True

    )


    print(
        "Original records:",
        len(y)
    )


    print(
        "Records used:",
        len(y_valid)
    )

    print()


    print(
        "Classes:"
    )


    print(

        y_valid

        .value_counts()

        .to_string()

    )

    print()


    # --------------------------------------------------------
    # LABEL ENCODING
    # --------------------------------------------------------

    label_encoder = LabelEncoder()


    y_encoded = (

        label_encoder.fit_transform(

            y_valid

        )

    )


    class_names = (

        label_encoder.classes_

    )


    print(
        "Encoded classes:"
    )


    for number, class_name in enumerate(

        class_names

    ):

        print(

            number,

            "->",

            class_name

        )


    print()


    # --------------------------------------------------------
    # TRAIN / TEST SPLIT
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = (

        train_test_split(

            X_valid,

            y_encoded,

            test_size=TEST_SIZE,

            random_state=RANDOM_STATE,

            stratify=y_encoded

        )

    )


    print(
        "Training samples:",
        len(X_train)
    )


    print(
        "Testing samples:",
        len(X_test)
    )

    print()


    # --------------------------------------------------------
    # CLASSIFIER
    # --------------------------------------------------------

    classifier = LogisticRegression(

        max_iter=2000,

        random_state=RANDOM_STATE,

        class_weight="balanced"

    )


    print(
        "Training Logistic Regression..."
    )

    print()


    classifier.fit(

        X_train,

        y_train

    )


    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    y_pred = classifier.predict(

        X_test

    )


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    accuracy = accuracy_score(

        y_test,

        y_pred

    )


    macro_f1 = f1_score(

        y_test,

        y_pred,

        average="macro",

        zero_division=0

    )


    weighted_f1 = f1_score(

        y_test,

        y_pred,

        average="weighted",

        zero_division=0

    )


    print()
    print("=" * 70)

    print(
        f"RESULTS: {target_name}"
    )

    print("=" * 70)
    print()


    print(
        f"Accuracy:       {accuracy:.4f}"
    )


    print(
        f"Macro F1:       {macro_f1:.4f}"
    )


    print(
        f"Weighted F1:    {weighted_f1:.4f}"
    )

    print()


    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    report = classification_report(

        y_test,

        y_pred,

        target_names=class_names,

        output_dict=True,

        zero_division=0

    )


    report_df = pd.DataFrame(

        report

    ).transpose()


    print(
        "CLASSIFICATION REPORT"
    )

    print()


    print(

        report_df.to_string()

    )

    print()


    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(

        y_test,

        y_pred

    )


    print(
        "CONFUSION MATRIX"
    )

    print()


    print(
        cm
    )

    print()


    # --------------------------------------------------------
    # PLOT CONFUSION MATRIX
    # --------------------------------------------------------

    plt.figure(

        figsize=(

            max(

                8,

                len(class_names) * 1.2

            ),

            max(

                6,

                len(class_names) * 1.0

            )

        )

    )


    sns.heatmap(

        cm,

        annot=True,

        fmt="d",

        cmap="Blues",

        xticklabels=class_names,

        yticklabels=class_names

    )


    plt.xlabel(
        "Predicted"
    )


    plt.ylabel(
        "Actual"
    )


    plt.title(
        f"{target_name} Classification"
    )


    plt.tight_layout()


    plt.savefig(

        confusion_matrix_file,

        dpi=300,

        bbox_inches="tight"

    )


    plt.close()


    print(
        "Confusion matrix saved:",
        confusion_matrix_file
    )


    # --------------------------------------------------------
    # SAVE TEST PREDICTIONS
    # --------------------------------------------------------

    prediction_rows = []


    valid_original_indices = (

        np.where(

            valid_mask.values

        )[0]

    )


    _, test_indices = train_test_split(

        np.arange(

            len(y_valid)

        ),

        test_size=TEST_SIZE,

        random_state=RANDOM_STATE,

        stratify=y_encoded

    )


    for position, actual, predicted in zip(

        test_indices,

        y_test,

        y_pred

    ):

        original_index = (

            valid_original_indices[position]

        )


        prediction_rows.append({

            "Original_Row":
                original_index + 2,

            "Actual":
                class_names[actual],

            "Predicted":
                class_names[predicted],

            "Correct":
                actual == predicted

        })


    prediction_df = pd.DataFrame(

        prediction_rows

    )


    # --------------------------------------------------------
    # SAVE RESULTS TO EXCEL
    # --------------------------------------------------------

    with pd.ExcelWriter(

        excel_file,

        engine="openpyxl"

    ) as writer:


        report_df.to_excel(

            writer,

            sheet_name="Classification_Report"

        )


        pd.DataFrame({

            "Metric": [

                "Accuracy",

                "Macro F1",

                "Weighted F1"

            ],

            "Value": [

                accuracy,

                macro_f1,

                weighted_f1

            ]

        }).to_excel(

            writer,

            sheet_name="Overall_Metrics",

            index=False

        )


        cm_df = pd.DataFrame(

            cm,

            index=class_names,

            columns=class_names

        )


        cm_df.to_excel(

            writer,

            sheet_name="Confusion_Matrix"

        )


        prediction_df.to_excel(

            writer,

            sheet_name="Test_Predictions",

            index=False

        )


    print(
        "Classifier results saved:",
        excel_file
    )

    print()


    return {

        "accuracy":
            accuracy,

        "macro_f1":
            macro_f1,

        "weighted_f1":
            weighted_f1,

        "classifier":
            classifier,

        "label_encoder":
            label_encoder

    }


# ============================================================
# TRAIN TRIAGE CLASSIFIER
# ============================================================

triage_results = train_classifier(

    fused_features,

    triage_levels,

    "TRIAGE LEVEL",

    TRIAGE_CONFUSION_MATRIX,

    TRIAGE_MODEL_RESULTS

)


# ============================================================
# TRAIN DEPARTMENT CLASSIFIER
# ============================================================

department_results = train_classifier(

    fused_features,

    departments,

    "DEPARTMENT",

    DEPARTMENT_CONFUSION_MATRIX,

    DEPARTMENT_MODEL_RESULTS

)


# ============================================================
# FINAL COMPARISON
# ============================================================

print()
print("=" * 70)
print("FINAL COMPARISON")
print("=" * 70)
print()


print(
    "TRIAGE LEVEL CLASSIFIER"
)

print(
    f"Accuracy:    "
    f"{triage_results['accuracy']:.4f}"
)

print(
    f"Macro F1:    "
    f"{triage_results['macro_f1']:.4f}"
)

print(
    f"Weighted F1: "
    f"{triage_results['weighted_f1']:.4f}"
)

print()


print(
    "DEPARTMENT CLASSIFIER"
)

print(
    f"Accuracy:    "
    f"{department_results['accuracy']:.4f}"
)

print(
    f"Macro F1:    "
    f"{department_results['macro_f1']:.4f}"
)

print(
    f"Weighted F1: "
    f"{department_results['weighted_f1']:.4f}"
)

print()


# ============================================================
# FINISHED
# ============================================================

print("=" * 70)
print("EXPERIMENT FINISHED")
print("=" * 70)
print()


print(
    "Input file:",
    CSV_FILE
)


print(
    "Embedding model:",
    EMBEDDING_MODEL
)


print(
    "Translation model:",
    MODEL
)


print(
    "Total records:",
    len(results)
)


print(
    "API translations performed:",
    len(translations)
)


print(
    "Fused feature dimensions:",
    fused_features.shape[1]
)

print()


print(
    "Processed dataset:",
    PROCESSED_DATA_FILE
)


print(
    "Embeddings:",
    EMBEDDING_FILE
)


print(
    "Triage results:",
    TRIAGE_MODEL_RESULTS
)


print(
    "Department results:",
    DEPARTMENT_MODEL_RESULTS
)


print(
    "Triage confusion matrix:",
    TRIAGE_CONFUSION_MATRIX
)


print(
    "Department confusion matrix:",
    DEPARTMENT_CONFUSION_MATRIX
)

print()