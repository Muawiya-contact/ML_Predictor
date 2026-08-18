import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from itertools import combinations

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# ============================================================
# SETTINGS
# ============================================================

CSV_FILE = "dataset.csv"


# ------------------------------------------------------------
# SENTENCE TRANSFORMER
# ------------------------------------------------------------

EMBEDDING_MODEL = "intfloat/multilingual-e5-small"


# ------------------------------------------------------------
# MINIMUM CLUSTER SIZE
# ------------------------------------------------------------

# A department is considered a valid cluster only if
# it contains at least this many complaints.

MIN_CLUSTER_SIZE = 25


# ------------------------------------------------------------
# OUTPUT FILES
# ------------------------------------------------------------

EMBEDDING_OUTPUT = "complaint_embeddings.npy"

SIMILARITY_OUTPUT = "pairwise_similarity_results.xlsx"

PCA_OUTPUT = "embedding_pca_2d.png"


# ============================================================
# READ DATASET
# ============================================================

print()
print("=" * 70)
print("READING DATASET")
print("=" * 70)
print()

try:

    df = pd.read_csv(CSV_FILE)

except Exception as e:

    print("Could not read dataset.csv")
    print(e)

    exit()


print(
    "Total records in dataset:",
    len(df)
)

print()


# ============================================================
# READ COLUMN D AND COLUMN M
# ============================================================

# Python column numbering starts from 0
#
# Column D = index 3
# Column M = index 12

chief_complaints = (
    df.iloc[:, 3]
    .fillna("")
    .astype(str)
)

departments = (
    df.iloc[:, 11]
    .fillna("")
    .astype(str)
)


# ============================================================
# REMOVE EMPTY RECORDS
# ============================================================

valid_mask = (
    (chief_complaints.str.strip() != "") &
    (departments.str.strip() != "")
)


chief_complaints = (
    chief_complaints[valid_mask]
    .reset_index(drop=True)
)

departments = (
    departments[valid_mask]
    .reset_index(drop=True)
)


print(
    "Valid complaints:",
    len(chief_complaints)
)

print()


# ============================================================
# COUNT ALL DEPARTMENTS
# ============================================================

print("=" * 70)
print("DEPARTMENT COUNTS")
print("=" * 70)
print()


department_counts = departments.value_counts()


for department, count in department_counts.items():

    if count >= MIN_CLUSTER_SIZE:

        status = "VALID CLUSTER"

    else:

        status = "EXCLUDED"


    print(
        f"{department}: {count} complaints --> {status}"
    )


print()


# ============================================================
# SELECT VALID CLUSTERS
# ============================================================

valid_departments = department_counts[
    department_counts >= MIN_CLUSTER_SIZE
].index


print("=" * 70)
print(
    f"VALID CLUSTERS (MINIMUM {MIN_CLUSTER_SIZE} COMPLAINTS)"
)
print("=" * 70)
print()


for department in valid_departments:

    print(
        f"{department}: "
        f"{department_counts[department]} complaints"
    )


print()


print(
    "Number of valid clusters:",
    len(valid_departments)
)


# ============================================================
# STOP IF NO VALID CLUSTERS
# ============================================================

if len(valid_departments) == 0:

    print()
    print("=" * 70)
    print("ERROR: NO VALID CLUSTERS")
    print("=" * 70)
    print()

    print(
        f"No department contains at least "
        f"{MIN_CLUSTER_SIZE} complaints."
    )

    print()
    print(
        "Actual department counts:"
    )

    print(
        department_counts
    )

    exit()


# ============================================================
# FILTER DATASET
# ============================================================

cluster_mask = departments.isin(
    valid_departments
)


chief_complaints = (
    chief_complaints[cluster_mask]
    .reset_index(drop=True)
)

departments = (
    departments[cluster_mask]
    .reset_index(drop=True)
)


print()
print(
    "Complaints retained for analysis:",
    len(chief_complaints)
)

print()


# ============================================================
# CREATE RESULT DATAFRAME
# ============================================================

results = pd.DataFrame({

    "Complaint_ID":
        range(
            1,
            len(chief_complaints) + 1
        ),

    "Roman_Urdu_Complaint":
        chief_complaints,

    "Department":
        departments

})


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
# GENERATE EMBEDDINGS
# ============================================================

print()
print("=" * 70)
print("GENERATING EMBEDDINGS FROM ROMAN URDU")
print("=" * 70)
print()


# IMPORTANT:
# The original Roman Urdu complaints are fed directly
# into the Sentence Transformer.
#
# There is NO GPT translation step.

texts = results[
    "Roman_Urdu_Complaint"
].tolist()


embeddings = embedding_model.encode(

    texts,

    convert_to_numpy=True,

    normalize_embeddings=True,

    show_progress_bar=True
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

    EMBEDDING_OUTPUT,

    embeddings
)


print()

print(
    "Embeddings saved:",
    EMBEDDING_OUTPUT
)


# ============================================================
# 2D PCA VISUALIZATION
# ============================================================

print()
print("=" * 70)
print("CREATING 2D PCA VISUALIZATION")
print("=" * 70)
print()


pca = PCA(
    n_components=2
)


embeddings_2d = pca.fit_transform(
    embeddings
)


explained_variance = (
    pca.explained_variance_ratio_
)


print(
    f"PC1 explained variance: "
    f"{explained_variance[0] * 100:.2f}%"
)


print(
    f"PC2 explained variance: "
    f"{explained_variance[1] * 100:.2f}%"
)


print(
    f"Total explained variance: "
    f"{sum(explained_variance) * 100:.2f}%"
)


# ------------------------------------------------------------
# CREATE PLOT
# ------------------------------------------------------------

plt.figure(
    figsize=(12, 9)
)


unique_departments = sorted(
    results["Department"].unique()
)


for department in unique_departments:

    mask = (
        results["Department"] == department
    ).values


    plt.scatter(

        embeddings_2d[mask, 0],

        embeddings_2d[mask, 1],

        label=department,

        alpha=0.75,

        s=50
    )


# ------------------------------------------------------------
# LABEL EACH POINT
# ------------------------------------------------------------

for i in range(
    len(results)
):

    plt.annotate(

        str(i + 1),

        (
            embeddings_2d[i, 0],

            embeddings_2d[i, 1]
        ),

        xytext=(4, 4),

        textcoords="offset points",

        fontsize=7
    )


plt.xlabel(
    "Principal Component 1"
)


plt.ylabel(
    "Principal Component 2"
)


plt.title(
    "2D PCA Projection of Sentence Transformer Embeddings\n"
    "Direct Roman Urdu Input"
)


plt.legend(

    title="Department",

    bbox_to_anchor=(1.05, 1),

    loc="upper left"
)


plt.grid(
    True,
    alpha=0.3
)


plt.tight_layout()


plt.savefig(

    PCA_OUTPUT,

    dpi=300,

    bbox_inches="tight"
)


plt.show()


print()

print(
    "2D plot saved:",
    PCA_OUTPUT
)


# ============================================================
# COSINE SIMILARITY MATRIX
# ============================================================

# Because embeddings were normalized,
# dot product = cosine similarity.

similarity_matrix = np.matmul(

    embeddings,

    embeddings.T
)


# ============================================================
# CALCULATE DEPARTMENT-WISE INTRA-CLUSTER SIMILARITY
# ============================================================

print()
print("=" * 70)
print("CALCULATING INTRA-DEPARTMENT SIMILARITY")
print("=" * 70)
print()


intra_results = []


unique_departments = sorted(
    results["Department"].unique()
)


for department in unique_departments:

    indices = results.index[
        results["Department"] == department
    ].tolist()


    if len(indices) < 2:

        mean_similarity = np.nan

        median_similarity = np.nan

        minimum_similarity = np.nan

        maximum_similarity = np.nan


    else:

        values = []


        for i, j in combinations(

            indices,

            2

        ):

            values.append(
                similarity_matrix[i, j]
            )


        mean_similarity = np.mean(
            values
        )


        median_similarity = np.median(
            values
        )


        minimum_similarity = np.min(
            values
        )


        maximum_similarity = np.max(
            values
        )


    intra_results.append({

        "Department":
            department,

        "Number_of_Complaints":
            len(indices),

        "Mean_Intra_Similarity":
            mean_similarity,

        "Median_Intra_Similarity":
            median_similarity,

        "Minimum_Intra_Similarity":
            minimum_similarity,

        "Maximum_Intra_Similarity":
            maximum_similarity

    })


intra_df = pd.DataFrame(
    intra_results
)


# ============================================================
# SAME-DEPARTMENT AND DIFFERENT-DEPARTMENT PAIRS
# ============================================================

same_cluster_values = []

different_cluster_values = []


for i in range(
    len(results)
):

    for j in range(

        i + 1,

        len(results)

    ):

        similarity = similarity_matrix[
            i,
            j
        ]


        if (

            departments.iloc[i]

            ==

            departments.iloc[j]

        ):

            same_cluster_values.append(
                similarity
            )

        else:

            different_cluster_values.append(
                similarity
            )


# ============================================================
# CHECK THAT BOTH TYPES OF PAIRS EXIST
# ============================================================

if len(same_cluster_values) == 0:

    print(
        "ERROR: No same-department pairs found."
    )

    exit()


if len(different_cluster_values) == 0:

    print(
        "ERROR: No different-department pairs found."
    )

    exit()


# ============================================================
# GLOBAL STATISTICS
# ============================================================

mean_same = np.mean(
    same_cluster_values
)


median_same = np.median(
    same_cluster_values
)


min_same = np.min(
    same_cluster_values
)


max_same = np.max(
    same_cluster_values
)


mean_different = np.mean(
    different_cluster_values
)


median_different = np.median(
    different_cluster_values
)


min_different = np.min(
    different_cluster_values
)


max_different = np.max(
    different_cluster_values
)


# ============================================================
# SEMANTIC SEPARATION
# ============================================================

semantic_separation = (

    mean_same

    -

    mean_different
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print()
print("=" * 70)
print("SEMANTIC SIMILARITY RESULTS")
print("=" * 70)
print()


print(
    f"Mean similarity - SAME department: "
    f"{mean_same:.4f}"
)


print(
    f"Mean similarity - DIFFERENT departments: "
    f"{mean_different:.4f}"
)


print()


print(
    f"Median similarity - SAME department: "
    f"{median_same:.4f}"
)


print(
    f"Median similarity - DIFFERENT departments: "
    f"{median_different:.4f}"
)


print()


print(
    f"Minimum intra-department similarity: "
    f"{min_same:.4f}"
)


print(
    f"Maximum intra-department similarity: "
    f"{max_same:.4f}"
)


print()


print(
    f"Minimum inter-department similarity: "
    f"{min_different:.4f}"
)


print(
    f"Maximum inter-department similarity: "
    f"{max_different:.4f}"
)


print()


print(
    f"Semantic separation: "
    f"{semantic_separation:.4f}"
)


# ============================================================
# DEPARTMENT-WISE RESULTS
# ============================================================

print()
print("=" * 70)
print("DEPARTMENT-WISE RESULTS")
print("=" * 70)
print()


print(
    intra_df.to_string(
        index=False
    )
)


# ============================================================
# CREATE PAIRWISE RESULTS
# ============================================================

print()
print("=" * 70)
print("CREATING PAIRWISE SIMILARITY RESULTS")
print("=" * 70)
print()


pairwise_rows = []


for i in range(
    len(results)
):

    for j in range(

        i + 1,

        len(results)

    ):

        same_department = (

            departments.iloc[i]

            ==

            departments.iloc[j]

        )


        pairwise_rows.append({

            "Complaint_1":
                i + 1,

            "Department_1":
                departments.iloc[i],

            "Roman_Urdu_1":
                chief_complaints.iloc[i],

            "Complaint_2":
                j + 1,

            "Department_2":
                departments.iloc[j],

            "Roman_Urdu_2":
                chief_complaints.iloc[j],

            "Cosine_Similarity":
                similarity_matrix[i, j],

            "Same_Department":
                same_department

        })


pairwise_df = pd.DataFrame(
    pairwise_rows
)


# ============================================================
# SAVE EVERYTHING TO EXCEL
# ============================================================

with pd.ExcelWriter(

    SIMILARITY_OUTPUT,

    engine="openpyxl"

) as writer:


    # --------------------------------------------------------
    # DEPARTMENT STATISTICS
    # --------------------------------------------------------

    intra_df.to_excel(

        writer,

        sheet_name="Department_Statistics",

        index=False
    )


    # --------------------------------------------------------
    # PAIRWISE SIMILARITY
    # --------------------------------------------------------

    pairwise_df.to_excel(

        writer,

        sheet_name="Pairwise_Similarity",

        index=False
    )


    # --------------------------------------------------------
    # ORIGINAL COMPLAINTS
    # --------------------------------------------------------

    results.to_excel(

        writer,

        sheet_name="Complaints",

        index=False
    )


# ============================================================
# FINISHED
# ============================================================

print()
print("=" * 70)
print("FINISHED")
print("=" * 70)
print()


print(
    "Valid clusters:",
    len(valid_departments)
)


print(
    "Minimum cluster size:",
    MIN_CLUSTER_SIZE
)


print(
    "Complaints analyzed:",
    len(results)
)


print()


print(
    "Embedding file:",
    EMBEDDING_OUTPUT
)


print(
    "Similarity file:",
    SIMILARITY_OUTPUT
)


print(
    "PCA plot:",
    PCA_OUTPUT
)


print()


print(
    "Semantic separation:",
    f"{semantic_separation:.4f}"
)


print()