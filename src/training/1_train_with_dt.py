import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# ==============================
# CONFIG
# ==============================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATASET_DIR = os.path.join(BASE_DIR, "4_dataset_POINT") 
# If you want to train it on a non augmented dataest use this directory
# DATASET_DIR = os.path.join(BASE_DIR, "4_1_dataset_unprocessed") 
OUTPUT_DIR = os.path.join(BASE_DIR, "6_results")
OUTPUT_MODEL = os.path.join(BASE_DIR, "model")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_NAMES = [
    "bird", "boar", "dog", "dragon", "hare", "horse",
    "monkey", "ox", "ram", "rat", "snake", "tiger"
]

# ==============================
# LOAD DATA
# ==============================

def load_split(split_name):
    path = os.path.join(DATASET_DIR, split_name, f"{split_name}.csv")
    df   = pd.read_csv(path)
    X    = df.drop(columns=[c for c in ["person_id", "class_id", "label"] if c in df.columns]).values
    y    = df["class_id"].values
    return X, y, df

print("Loading data...")
X_train, y_train, df_train = load_split("train")
X_val,   y_val,   df_val   = load_split("validation")
X_test,  y_test,  df_test  = load_split("test")

feature_names = df_train.drop(columns=["person_id", "class_id"]).columns.tolist()

print(f"  train      : {X_train.shape[0]} rows")
print(f"  validation : {X_val.shape[0]} rows")
print(f"  test       : {X_test.shape[0]} rows")

# ==============================
# EVALUATE HELPER
# ==============================

def evaluate(clf, X, y, split_name):
    y_pred = clf.predict(X)
    acc    = accuracy_score(y, y_pred)
    print(f"  {split_name:<12} accuracy: {acc:.1%}")
    return y_pred

# ==============================
# DEPTH VS VALIDATION ACCURACY
# ==============================

print("\nComputing depth vs accuracy curve...")
depths     = list(range(1, 31))
train_accs = []
val_accs   = []

for d in depths:
    t = DecisionTreeClassifier(max_depth=d, min_samples_leaf=5, random_state=42)
    t.fit(X_train, y_train)
    train_accs.append(accuracy_score(y_train, t.predict(X_train)))
    val_accs.append(  accuracy_score(y_val,   t.predict(X_val)))

best_depth = depths[int(np.argmax(val_accs))]
print(f"Best depth = {best_depth}")

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(depths, train_accs, label="train",      marker='o', markersize=4)
ax.plot(depths, val_accs,   label="validation", marker='o', markersize=4)
ax.axvline(best_depth, color='gray', linestyle='--',
           label=f"best depth = {best_depth}")
ax.set_xlabel("Max depth")
ax.set_ylabel("Accuracy")
ax.set_title("Decision Tree — depth vs accuracy")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
curve_path = os.path.join(OUTPUT_DIR, "dt_depth_curve.png")
plt.savefig(curve_path, dpi=150)
plt.close()
print(f"Depth curve saved      → {curve_path}")

# ==============================
# RETRAIN AT BEST DEPTH
# ==============================

print(f"\nRetraining at best depth = {best_depth}...")
clf_best = DecisionTreeClassifier(
    max_depth=best_depth,
    min_samples_leaf=5,
    random_state=42
)
clf_best.fit(X_train, y_train)

print("\nFinal accuracy (best depth model):")
evaluate(clf_best, X_train, y_train, "train")
evaluate(clf_best, X_val,   y_val,   "validation")
y_test_pred = evaluate(clf_best, X_test, y_test, "test")

# ==============================
# CONFUSION MATRIX
# ==============================

cm = confusion_matrix(y_test, y_test_pred)
fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
disp.plot(ax=ax, colorbar=True, xticks_rotation=45)
ax.set_title("Confusion matrix — test split")
plt.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix_dt.png")
plt.savefig(cm_path, dpi=150)
plt.close()
print(f"Confusion matrix saved → {cm_path}")

# ==============================
# FEATURE IMPORTANCE
# ==============================

print("\nComputing feature importance...")
importances = clf_best.feature_importances_
indices     = np.argsort(importances)[::-1][:20]  # top 20 features

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(range(20), importances[indices], color='steelblue', alpha=0.8)
ax.set_xticks(range(20))
ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right', fontsize=8)
ax.set_ylabel("Importance")
ax.set_title("Decision Tree — top 20 most important features")
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
fi_path = os.path.join(OUTPUT_DIR, "feature_importance_dt.png")
plt.savefig(fi_path, dpi=150)
plt.close()
print(f"Feature importance saved → {fi_path}")

# ==============================
# SAVE MODEL
# ==============================

# Use the proper filename for the model depending on the dataset you trained on:
# model_path = os.path.join(OUTPUT_MODEL, "test_arham.joblib")  # for augmented dataset
model_path = os.path.join(OUTPUT_MODEL, "model_dt.joblib")
joblib.dump({
    "model":         clf_best,
    "class_names":   CLASS_NAMES,
    "best_depth":    best_depth,
    "feature_names": feature_names,
    "train_accs":    train_accs,
    "val_accs":      val_accs,
}, model_path)

print(f"Model saved            → {model_path}")
print("\nDone.")