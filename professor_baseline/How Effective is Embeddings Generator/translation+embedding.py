import os  # added for the redacted API key
import requests
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# SETTINGS
# ============================================================

API_KEY = os.environ.get("OPENROUTER_API_KEY", "")  # REDACTED: a live key was hardcoded here

MODEL = "openai/gpt-4o-mini"

URL = "https://openrouter.ai/api/v1/chat/completions"

INPUT_FILE = "ground_truth_clusters.xlsx"

OUTPUT_FILE = "embedding_similarity_results.xlsx"

# Use the Sentence Transformer you already have installed.
# Change this if your existing model is different.
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"


# ============================================================
# LOAD SENTENCE TRANSFORMER
# ============================================================

print()
print("=" * 70)
print("Loading Sentence Transformer")
print("=" * 70)

embedding_model = SentenceTransformer(EMBEDDING_MODEL)

print("Model:", EMBEDDING_MODEL)
print("Model loaded successfully.")
print()


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
5. Correct obvious Roman Urdu spelling variations when necessary.
6. Return ONLY the English translation.
7. Do not provide explanations.

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

    try:

        response = requests.post(
            URL,
            headers=headers,
            json=data,
            timeout=60
        )

    except Exception as e:

        print("Connection error:")
        print(e)
        return None


    if response.status_code != 200:

        print()
        print("API Error")
        print("Status:", response.status_code)
        print(response.text)

        return None


    try:

        result = response.json()

    except Exception as e:

        print("Could not read API response:")
        print(e)

        return None


    try:

        choices = result.get("choices")

        if not choices:
            print("No choices returned.")
            return None

        message = choices[0].get("message")

        if not message:
            print("No message returned.")
            return None

        translation = message.get("content")

        if translation is None:
            print("API returned no translation.")
            return None

        return translation.strip()

    except Exception as e:

        print("Error extracting translation:")
        print(e)

        return None


# ============================================================
# READ EXCEL FILE
# ============================================================

print("=" * 70)
print("Reading ground_truth_clusters.xlsx")
print("=" * 70)

df = pd.read_excel(INPUT_FILE)

print()
print("Columns found:")
print(df.columns.tolist())
print()


# ============================================================
# USE FIRST TWO COLUMNS
# ============================================================

cluster1 = df.iloc[:, 0].dropna().astype(str).tolist()
cluster2 = df.iloc[:, 1].dropna().astype(str).tolist()

print("Cluster 1 complaints:", len(cluster1))
print("Cluster 2 complaints:", len(cluster2))
print()


# ============================================================
# TRANSLATE ALL COMPLAINTS
# ============================================================

all_complaints = cluster1 + cluster2

print("=" * 70)
print("TRANSLATING COMPLAINTS")
print("=" * 70)

translations = []

for i, complaint in enumerate(all_complaints):

    print(
        f"[{i + 1}/{len(all_complaints)}] "
        f"{complaint}"
    )

    translation = translate_roman_urdu(complaint)

    if translation is None:

        print("Translation failed.")

        translation = ""

    print("English:", translation)
    print()

    translations.append(translation)


# ============================================================
# SEPARATE TRANSLATIONS
# ============================================================

translations_cluster1 = translations[:len(cluster1)]
translations_cluster2 = translations[len(cluster1):]


# ============================================================
# GENERATE EMBEDDINGS
# ============================================================

print("=" * 70)
print("GENERATING EMBEDDINGS")
print("=" * 70)

all_translations = (
    translations_cluster1 +
    translations_cluster2
)

embeddings = embedding_model.encode(
    all_translations,
    normalize_embeddings=True,
    show_progress_bar=True
)

embeddings = np.asarray(embeddings)

print()
print("Embedding shape:", embeddings.shape)
print()


# ============================================================
# SPLIT EMBEDDINGS
# ============================================================

n1 = len(cluster1)

embeddings_cluster1 = embeddings[:n1]

embeddings_cluster2 = embeddings[n1:]


# ============================================================
# COSINE SIMILARITY MATRICES
# ============================================================

print("=" * 70)
print("CALCULATING COSINE SIMILARITIES")
print("=" * 70)


# ------------------------------------------------------------
# Within Cluster 1
# ------------------------------------------------------------

similarity_c1 = cosine_similarity(
    embeddings_cluster1,
    embeddings_cluster1
)


# ------------------------------------------------------------
# Within Cluster 2
# ------------------------------------------------------------

similarity_c2 = cosine_similarity(
    embeddings_cluster2,
    embeddings_cluster2
)


# ------------------------------------------------------------
# Between Cluster 1 and Cluster 2
# ------------------------------------------------------------

similarity_between = cosine_similarity(
    embeddings_cluster1,
    embeddings_cluster2
)


# ============================================================
# REMOVE SELF-SIMILARITY FROM INTRA-CLUSTER MATRICES
# ============================================================

# Diagonal values are always 1 because every sentence is
# identical to itself.

np.fill_diagonal(
    similarity_c1,
    np.nan
)

np.fill_diagonal(
    similarity_c2,
    np.nan
)


# ============================================================
# EXTRACT INTRA-CLUSTER VALUES
# ============================================================

intra_c1_values = similarity_c1[
    ~np.isnan(similarity_c1)
]

intra_c2_values = similarity_c2[
    ~np.isnan(similarity_c2)
]


# ============================================================
# INTER-CLUSTER VALUES
# ============================================================

inter_values = similarity_between.flatten()


# ============================================================
# STATISTICS
# ============================================================

mean_intra_c1 = np.mean(intra_c1_values)

mean_intra_c2 = np.mean(intra_c2_values)

mean_intra = np.mean(
    np.concatenate(
        [intra_c1_values, intra_c2_values]
    )
)

mean_inter = np.mean(inter_values)


median_intra = np.median(
    np.concatenate(
        [intra_c1_values, intra_c2_values]
    )
)

median_inter = np.median(inter_values)


min_intra = np.min(
    np.concatenate(
        [intra_c1_values, intra_c2_values]
    )
)

max_intra = np.max(
    np.concatenate(
        [intra_c1_values, intra_c2_values]
    )
)

min_inter = np.min(inter_values)

max_inter = np.max(inter_values)


# ============================================================
# SEMANTIC SEPARATION
# ============================================================

semantic_separation = (
    mean_intra - mean_inter
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 70)
print("SEMANTIC SIMILARITY RESULTS")
print("=" * 70)

print()

print(
    f"Mean similarity - Cluster 1: "
    f"{mean_intra_c1:.4f}"
)

print(
    f"Mean similarity - Cluster 2: "
    f"{mean_intra_c2:.4f}"
)

print(
    f"Mean similarity - SAME cluster: "
    f"{mean_intra:.4f}"
)

print(
    f"Mean similarity - DIFFERENT clusters: "
    f"{mean_inter:.4f}"
)

print()

print(
    f"Median similarity - SAME cluster: "
    f"{median_intra:.4f}"
)

print(
    f"Median similarity - DIFFERENT clusters: "
    f"{median_inter:.4f}"
)

print()

print(
    f"Minimum intra-cluster similarity: "
    f"{min_intra:.4f}"
)

print(
    f"Maximum intra-cluster similarity: "
    f"{max_intra:.4f}"
)

print()

print(
    f"Minimum inter-cluster similarity: "
    f"{min_inter:.4f}"
)

print(
    f"Maximum inter-cluster similarity: "
    f"{max_inter:.4f}"
)

print()

print(
    f"Semantic separation: "
    f"{semantic_separation:.4f}"
)

print()


# ============================================================
# SAVE TRANSLATIONS
# ============================================================

translation_rows = []

for i, complaint in enumerate(cluster1):

    translation_rows.append([
        "Cluster 1",
        complaint,
        translations_cluster1[i]
    ])


for i, complaint in enumerate(cluster2):

    translation_rows.append([
        "Cluster 2",
        complaint,
        translations_cluster2[i]
    ])


translations_df = pd.DataFrame(
    translation_rows,
    columns=[
        "Cluster",
        "Roman_Urdu",
        "English_Translation"
    ]
)


# ============================================================
# SAVE PAIRWISE SIMILARITIES
# ============================================================

pair_rows = []


# ------------------------------------------------------------
# Cluster 1 vs Cluster 1
# ------------------------------------------------------------

for i in range(len(cluster1)):

    for j in range(i + 1, len(cluster1)):

        pair_rows.append([
            "Cluster 1",
            cluster1[i],
            cluster1[j],
            similarity_c1[i, j]
        ])


# ------------------------------------------------------------
# Cluster 2 vs Cluster 2
# ------------------------------------------------------------

for i in range(len(cluster2)):

    for j in range(i + 1, len(cluster2)):

        pair_rows.append([
            "Cluster 2",
            cluster2[i],
            cluster2[j],
            similarity_c2[i, j]
        ])


# ------------------------------------------------------------
# Cluster 1 vs Cluster 2
# ------------------------------------------------------------

for i in range(len(cluster1)):

    for j in range(len(cluster2)):

        pair_rows.append([
            "Between Clusters",
            cluster1[i],
            cluster2[j],
            similarity_between[i, j]
        ])


pairs_df = pd.DataFrame(
    pair_rows,
    columns=[
        "Relationship",
        "Complaint_A",
        "Complaint_B",
        "Cosine_Similarity"
    ]
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_df = pd.DataFrame({

    "Measure": [
        "Number of Cluster 1 complaints",
        "Number of Cluster 2 complaints",
        "Embedding dimensions",
        "Mean intra-cluster similarity - Cluster 1",
        "Mean intra-cluster similarity - Cluster 2",
        "Mean intra-cluster similarity - Overall",
        "Mean inter-cluster similarity",
        "Median intra-cluster similarity",
        "Median inter-cluster similarity",
        "Minimum intra-cluster similarity",
        "Maximum intra-cluster similarity",
        "Minimum inter-cluster similarity",
        "Maximum inter-cluster similarity",
        "Semantic separation"
    ],

    "Value": [
        len(cluster1),
        len(cluster2),
        embeddings.shape[1],
        mean_intra_c1,
        mean_intra_c2,
        mean_intra,
        mean_inter,
        median_intra,
        median_inter,
        min_intra,
        max_intra,
        min_inter,
        max_inter,
        semantic_separation
    ]
})


# ============================================================
# SAVE EVERYTHING TO EXCEL
# ============================================================

print("=" * 70)
print("SAVING RESULTS")
print("=" * 70)

with pd.ExcelWriter(
    OUTPUT_FILE,
    engine="openpyxl"
) as writer:

    translations_df.to_excel(
        writer,
        sheet_name="Translations",
        index=False
    )

    pairs_df.to_excel(
        writer,
        sheet_name="Pairwise_Similarity",
        index=False
    )

    summary_df.to_excel(
        writer,
        sheet_name="Summary",
        index=False
    )


print()
print("Results saved to:")
print(OUTPUT_FILE)

print()
print("=" * 70)
print("DONE")
print("=" * 70)