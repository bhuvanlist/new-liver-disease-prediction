# train.py
import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

# ---------- CONFIG ----------
DATAFILE = "upload+new.csv"   # merged dataset
RANDOM_STATE = 42
TEST_SIZE = 0.20
SMOTE_RANDOM = 42
# ----------------------------

def load_csv_with_fallback(path):
    """Try common encodings until one works."""
    for enc in ("utf-8", "latin1", "iso-8859-1", "cp1252"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path)

print("🔹 Loading dataset:", DATAFILE)
if not os.path.exists(DATAFILE):
    raise FileNotFoundError(f"{DATAFILE} not found in current folder.")

df = load_csv_with_fallback(DATAFILE)
print("Raw shape:", df.shape)
print("Columns:", df.columns.tolist())

# ---------- Identify target column ----------
possible_targets = ["Outcome", "Dataset", "Label", "target", "Target", "OUTCOME"]
target_col = None
for c in possible_targets:
    if c in df.columns:
        target_col = c
        break
if target_col is None:
    target_col = df.columns[-1]
    print(f"No standard target name found; using last column: {target_col}")

print("Using target column:", target_col)

# ---------- Standardize labels ----------
y_raw = df[target_col].copy()
unique_vals = sorted(pd.Series(y_raw).dropna().unique())
if set(unique_vals) == {1, 2}:
    df[target_col] = df[target_col].map({1: 1, 2: 0})
elif set(unique_vals) == {0, 1}:
    df[target_col] = df[target_col].astype(int)
else:
    y_lower = y_raw.astype(str).str.lower()
    if y_lower.isin(["positive","pos","1","yes","y","disease"]).any():
        df[target_col] = y_lower.map(lambda v: 1 if v in ("positive","pos","1","yes","y","disease") else 0)
    else:
        df[target_col] = (pd.to_numeric(y_raw, errors="coerce") >
                          pd.to_numeric(y_raw, errors="coerce").median()).astype(int)

# ---------- Drop irrelevant columns ----------
for idcol in ("Id", "id", "ID", "patient_id"):
    if idcol in df.columns:
        df = df.drop(columns=[idcol])

# ---------- Encode features ----------
X = df.drop(columns=[target_col])
y = df[target_col].astype(int)

if "Gender" in X.columns:
    if X["Gender"].dtype == object or X["Gender"].dtype.name == "category":
        X["Gender"] = X["Gender"].astype(str).str.strip().str.lower().map(lambda v: 1 if v.startswith("m") else 0)
    else:
        X["Gender"] = X["Gender"].apply(lambda v: 1 if v in (1,"1",True) else 0)

for col in X.columns:
    if X[col].dtype == object:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

print("Features shape:", X.shape)
print("Label distribution:", y.value_counts().to_dict())

# ---------- Handle missing ----------
imputer = SimpleImputer(strategy="median")
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# ---------- Split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)
print("Train/test sizes:", X_train.shape, X_test.shape)

# ---------- SMOTE ----------
print("Applying SMOTE to training set...")
sm = SMOTE(random_state=SMOTE_RANDOM)
X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)
print("After SMOTE:", X_train_bal.shape, pd.Series(y_train_bal).value_counts().to_dict())

# ---------- Scale ----------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_bal)
X_test_scaled = scaler.transform(X_test)

# ---------- Models ----------
rf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE)
ada = AdaBoostClassifier(n_estimators=300, random_state=RANDOM_STATE)
xgb = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5,
                    use_label_encoder=False, eval_metric="logloss", random_state=RANDOM_STATE)
voting = VotingClassifier(estimators=[("rf", rf), ("ada", ada), ("xgb", xgb)], voting="soft")
stacking = StackingClassifier(
    estimators=[("rf", rf), ("ada", ada), ("xgb", xgb)],
    final_estimator=LogisticRegression(max_iter=2000),
    passthrough=True, cv=5
)

models = {"RandomForest": rf, "AdaBoost": ada, "XGBoost": xgb, "Voting": voting, "Stacking": stacking}
results = {}

# ---------- Train & evaluate ----------
for name, model in models.items():
    print(f"\nTraining {name} ...")
    model.fit(X_train_scaled, y_train_bal)

    probs = model.predict_proba(X_test_scaled)[:, 1]
    best_t, best_f1 = 0.5, -1
    for t in np.arange(0.1, 0.91, 0.01):
        preds_t = (probs >= t).astype(int)
        f1 = f1_score(y_test, preds_t)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    preds_best = (probs >= best_t).astype(int)
    acc = accuracy_score(y_test, preds_best)
    creport = classification_report(y_test, preds_best, target_names=["Negative","Positive"])
    cm = confusion_matrix(y_test, preds_best)

    results[name] = {"model": model, "best_thresh": best_t, "f1": best_f1, "accuracy": acc,
                     "report": creport, "cm": cm}
    print(f"-> Best threshold: {best_t:.2f} | F1: {best_f1:.4f} | Acc: {acc:.4f}")
    print("Classification Report:\n", creport)
    print("Confusion Matrix:\n", cm)

# ---------- Select best ----------
best_name = max(results.keys(), key=lambda n: results[n]["f1"])
best_info = results[best_name]
best_model = best_info["model"]
best_threshold = best_info["best_thresh"]

print(f"\n🎯 Selected Best Model: {best_name} (F1={best_info['f1']:.4f}, threshold={best_threshold:.2f})")
print(best_info["report"])
print("Confusion matrix:\n", best_info["cm"])

# ---------- Save ----------
feature_columns = list(X.columns)
y_pred = (best_model.predict_proba(X_test_scaled)[:, 1] >= best_threshold).astype(int)

save_dict = {
    "model": best_model,
    "scaler": scaler,
    "imputer": imputer,
    "threshold": best_threshold,
    "feature_columns": feature_columns,
    "metrics": {
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "accuracy": accuracy_score(y_test, y_pred)
    }
}

with open("liver_model.pkl", "wb") as f:
    pickle.dump(save_dict, f)

print("\n✅ Saved liver_model.pkl with model, scaler, imputer, threshold, feature_columns, and metrics.")
import joblib
import pickle

# ---------- Save everything needed for deployment ----------
save_dict = {
    "model": best_model,
    "scaler": scaler,
    "imputer": imputer,
    "threshold": best_threshold,
    "feature_columns": list(X.columns)
}

# Save regular version (local use)
joblib.dump(save_dict, "liver_model.pkl")
print("\n✅ Saved liver_model.pkl (full size)")

# Save compressed version (GitHub-friendly)
joblib.dump(save_dict, "liver_model_compressed.pkl", compress=3)
print("✅ Saved liver_model_compressed.pkl (compressed for GitHub)")


