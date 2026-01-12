#!/usr/bin/env python3
"""
remap_greedy_diversity.py

- Loads your existing model pickle (default: liver_cluster_model_autolabel.pkl)
- Recomputes cluster scores if needed
- Produces a remap that ensures at least one cluster is assigned to each target disease
  by picking the highest-scoring cluster per disease (greedy)
- Assigns remaining clusters to their highest-scoring disease
- Saves remapped model and example CSV (greedy_remap_examples.csv)
"""
import os
import pickle
from collections import Counter
import pandas as pd
import numpy as np

MODEL_IN = "liver_cluster_model_autolabel.pkl"   # adjust if different
MODEL_OUT = "liver_cluster_model_remapped_greedy.pkl"
EXAMPLE_CSV = "greedy_remap_examples.csv"
DATA_PATH = "upload+new.csv"
N_EXAMPLES_PER_LABEL = 20

TARGET_LABELS = ["Hepatitis", "Fatty Liver", "Fibrosis", "Cirrhosis"]

if not os.path.exists(MODEL_IN):
    raise SystemExit(f"Model file not found: {MODEL_IN}")

with open(MODEL_IN, "rb") as f:
    saved = pickle.load(f)

features = saved["features"]
kmeans = saved["cluster_model"]
scaler = saved["scaler"]
imputer = saved["imputer"]
thresholds = saved.get("thresholds_used", {})

# scoring function (same as used before)
def score_cluster(center, THRESHOLDS):
    s = {"Hepatitis": 0, "Fatty Liver": 0, "Fibrosis": 0, "Cirrhosis": 0}
    ALT = center.get("Alamine_Aminotransferase", 0)
    AST = center.get("Aspartate_Aminotransferase", 0)
    TB = center.get("Total_Bilirubin", 0)
    ALP = center.get("Alkaline_Phosphotase", 0)
    ALB = center.get("Albumin", 0)
    AGR = center.get("Albumin_and_Globulin_Ratio", 0)
    AGE = center.get("Age", 0)

    T = {
        "ALT_high": THRESHOLDS.get("ALT_high", 100),
        "AST_high": THRESHOLDS.get("AST_high", 100),
        "ALT_mild": THRESHOLDS.get("ALT_mild", 40),
        "AST_mild": THRESHOLDS.get("AST_mild", 40),
        "Bilirubin_high": THRESHOLDS.get("Bilirubin_high", 2.0),
        "ALP_high": THRESHOLDS.get("ALP_high", 200),
        "Albumin_low": THRESHOLDS.get("Albumin_low", 3.0),
        "AG_low": THRESHOLDS.get("AG_low", 0.9),
        "Age_high": THRESHOLDS.get("Age_high", 60)
    }

    # Hepatitis
    if (ALT >= T["ALT_high"] or AST >= T["AST_high"]):
        s["Hepatitis"] += 3
    if ALT > AST:
        s["Hepatitis"] += 1
    if TB >= T["Bilirubin_high"]:
        s["Hepatitis"] += 1

    # Fatty Liver
    if (ALT >= T["ALT_mild"] or AST >= T["AST_mild"]) and (ALT < T["ALT_high"] and AST < T["AST_high"]):
        s["Fatty Liver"] += 2
    if (TB < T["Bilirubin_high"]) and (ALP < T["ALP_high"]) and (ALB >= T["Albumin_low"]):
        s["Fatty Liver"] += 1

    # Fibrosis
    if (ALT >= T["ALT_mild"] or AST >= T["AST_mild"]):
        s["Fibrosis"] += 1
    if (ALB < 3.5 and ALB >= T["Albumin_low"]):
        s["Fibrosis"] += 1
    if (TB >= 1.2 and TB < T["Bilirubin_high"]):
        s["Fibrosis"] += 1

    # Cirrhosis
    if ALB < T["Albumin_low"]:
        s["Cirrhosis"] += 3
    if AGR < T["AG_low"]:
        s["Cirrhosis"] += 2
    if AST > ALT:
        s["Cirrhosis"] += 1
    if TB >= 1.5:
        s["Cirrhosis"] += 1

    """# Cancer
    if ALP >= T["ALP_high"]:
        s["Cancer"] += 3
    if ALB < T["Albumin_low"]:
        s["Cancer"] += 1
    if AGE >= T["Age_high"]:
        s["Cancer"] += 1
    if ALP >= T["ALP_high"] and TB >= T["Bilirubin_high"]:
        s["Cancer"] += 1"""

    return s

# Reconstruct centers & scores
centers_scaled = kmeans.cluster_centers_
centers_original = scaler.inverse_transform(centers_scaled)
centers_df = pd.DataFrame(centers_original, columns=features)

cluster_scores = {}
for cid in centers_df.index:
    center = centers_df.loc[cid].to_dict()
    scores = score_cluster(center, thresholds)
    cluster_scores[cid] = scores

# Greedy assignment: for each target label pick the cluster with highest score for that label (if tie, pick highest absolute score)
assigned = {}
available = set(cluster_scores.keys())

for label in TARGET_LABELS:
    # find best available cluster for this label
    best = None
    best_score = -1e9
    for cid in available:
        sc = cluster_scores[cid].get(label, 0)
        if sc > best_score:
            best = cid
            best_score = sc
    if best is not None:
        assigned[best] = label
        available.remove(best)

# Assign remaining clusters to their top scoring label
for cid in list(available):
    scores = cluster_scores[cid]
    # pick label with max score
    lab = max(scores.items(), key=lambda x: (x[1], x[0]))[0]
    assigned[cid] = lab

# Summary
print("Greedy remap assignment (cluster -> label):")
for cid in sorted(assigned.keys()):
    print(f"  {cid} -> {assigned[cid]} (scores={cluster_scores[cid]})")

print("\nCounts:", Counter(assigned.values()))

# Save new mapping into model copy
saved_new = saved.copy()
saved_new["disease_map"] = assigned
saved_new["greedy_remap_info"] = {
    "target_labels": TARGET_LABELS,
    "cluster_scores": cluster_scores
}

with open(MODEL_OUT, "wb") as f:
    pickle.dump(saved_new, f)
print(f"\nSaved greedy remapped model to: {MODEL_OUT}")

# Produce CSV examples for inspection
if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    # normalize gender as before
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].map({"Male":1,"Female":0})
    # prepare X -> original cluster ids with saved_new model artifacts
    X_imp = saved_new["imputer"].transform(df[features])
    X_scaled = saved_new["scaler"].transform(X_imp)
    orig_labels = saved_new["cluster_model"].predict(X_scaled).astype(int)
    df_examples = df.copy()
    df_examples["original_cluster"] = orig_labels
    df_examples["remapped_label"] = df_examples["original_cluster"].map(assigned)
    rows = []
    for lab in TARGET_LABELS:
        subset = df_examples[df_examples["remapped_label"] == lab]
        if subset.shape[0] == 0:
            continue
        sample = subset.sample(n=min(N_EXAMPLES_PER_LABEL, len(subset)), random_state=42)
        rows.append(sample)
    if rows:
        out = pd.concat(rows, axis=0)
        out.to_csv(EXAMPLE_CSV, index=False)
        print(f"Wrote example rows to {EXAMPLE_CSV} (up to {N_EXAMPLES_PER_LABEL} per label).")
    else:
        print("No example rows to write (dataset or mapping empty).")
else:
    print("Data file not found; skipping example CSV generation.")

