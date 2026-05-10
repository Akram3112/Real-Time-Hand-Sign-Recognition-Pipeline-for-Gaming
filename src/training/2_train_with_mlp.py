import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
)
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ==============================
# CONFIG
# ==============================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SPLIT_PATH = os.path.join(BASE_DIR, "4_dataset_POINT")
RESULTS_PATH = os.path.join(BASE_DIR, "6_results")
OUTPUT_MODEL = os.path.join(BASE_DIR, "model")

LABELS = [
    "bird", "boar", "dog", "dragon", "hare", "horse",
    "monkey", "ox", "ram", "rat", "snake", "tiger"
]

SEED = 42
EPOCHS = 100
BATCH_SIZE = 128
PATIENCE = 15
LR = 5e-4
WEIGHT_DECAY = 1e-4

os.makedirs(RESULTS_PATH, exist_ok=True)

np.random.seed(SEED)
torch.manual_seed(SEED)

# ==============================
# LOAD FRAME-LEVEL SPLITS
# Each row/frame becomes one training sample
# ==============================


def load_split(split_name):
    # Map split names to file names
    split_file_map = {
        "train": "train.csv",
        "val": "validation.csv",
        "validation": "validation.csv",
        "test": "test.csv"
    }

    if split_name not in split_file_map:
        raise ValueError(f"Unknown split name: {split_name}")

    split_file = split_file_map[split_name]
    split_path = os.path.join(SPLIT_PATH, split_name, split_file)

    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Missing split file: {split_path}")

    df = pd.read_csv(split_path)
    if df.empty:
        raise RuntimeError(f"Empty split file: {split_path}")

    # Extract features (all columns except person_id, class_id, and optional label)
    feat_cols = [c for c in df.columns if c not in ["person_id", "class_id", "label"]]
    X_split = df[feat_cols].values.astype(float)

    # Extract labels
    y_split = df["class_id"].values

    return X_split, y_split


X_train, y_train = load_split("train")
X_val, y_val = load_split("validation")
X_test, y_test = load_split("test")

print(f"Classes found: {LABELS}")

print("\nFrame-level split:")
print(f"  Train: {len(X_train)} frames")
print(f"  Val:   {len(X_val)} frames")
print(f"  Test:  {len(X_test)} frames")

# Warn if any class is missing from test
test_classes = set(np.unique(y_test))
missing = set(range(len(LABELS))) - test_classes
if missing:
    print(f"\nWarning: missing from test set: {[LABELS[i] for i in missing]}")
else:
    print("\nAll classes represented in test set.")

# ==============================
# SCALE (fit on train only)
# ==============================

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# ==============================
# CLASS WEIGHTS
# Helps with class imbalance
# ==============================

class_weights_np = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

# Need full-length weight vector if any class is absent in train
full_weights = np.ones(len(LABELS), dtype=np.float32)
present_classes = np.unique(y_train)
for cls_idx, w in zip(present_classes, class_weights_np):
    full_weights[cls_idx] = w

# ==============================
# DATALOADERS
# ==============================

def to_loader(X_arr, y_arr, shuffle=False):
    dataset = TensorDataset(
        torch.tensor(X_arr, dtype=torch.float32),
        torch.tensor(y_arr, dtype=torch.long),
    )
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)

train_loader = to_loader(X_train, y_train, shuffle=True)
val_loader = to_loader(X_val, y_val, shuffle=False)
test_loader = to_loader(X_test, y_test, shuffle=False)

# ==============================
# MODEL
# ==============================

class MudraMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.40),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_dim = X_train.shape[1]
num_classes = len(LABELS)

model = MudraMLP(input_dim=input_dim, num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss(
    weight=torch.tensor(full_weights, dtype=torch.float32, device=device)
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    patience=5,
    factor=0.5
)

print(f"\nInput dim: {input_dim}")
print(f"Device: {device}")

# ==============================
# TRAINING LOOP
# ==============================

best_val_loss = float("inf")
best_state = None
epochs_no_improve = 0

train_losses, val_losses = [], []
train_accs, val_accs = [], []

for epoch in range(1, EPOCHS + 1):
    # ---- train ----
    model.train()
    total_train_loss = 0.0
    total_train_correct = 0
    total_train_count = 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item() * len(xb)
        total_train_correct += (logits.argmax(dim=1) == yb).sum().item()
        total_train_count += len(xb)

    train_loss = total_train_loss / total_train_count
    train_acc = total_train_correct / total_train_count

    # ---- val ----
    model.eval()
    total_val_loss = 0.0
    total_val_correct = 0
    total_val_count = 0

    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)

            logits = model(xb)
            loss = criterion(logits, yb)

            total_val_loss += loss.item() * len(xb)
            total_val_correct += (logits.argmax(dim=1) == yb).sum().item()
            total_val_count += len(xb)

    val_loss = total_val_loss / total_val_count
    val_acc = total_val_correct / total_val_count

    scheduler.step(val_loss)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(
        f"Epoch {epoch:3d} | "
        f"train loss {train_loss:.4f} acc {train_acc:.2%} | "
        f"val loss {val_loss:.4f} acc {val_acc:.2%}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = copy.deepcopy(model.state_dict())
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

# ==============================
# TEST
# ==============================

if best_state is None:
    raise RuntimeError("Training finished without saving a best model state.")

model.load_state_dict(best_state)
model.eval()

def predict_loader(loader):
    preds_all = []
    labels_all = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            preds = logits.argmax(dim=1).cpu().numpy()

            preds_all.extend(preds)
            labels_all.extend(yb.numpy())

    return np.array(preds_all), np.array(labels_all)


train_preds, train_labels = predict_loader(train_loader)
val_preds, val_labels = predict_loader(val_loader)
all_preds, all_labels = predict_loader(test_loader)

class_names = LABELS

train_acc = (train_preds == train_labels).mean()
val_acc = (val_preds == val_labels).mean()
test_acc = (all_preds == all_labels).mean()

print("\nFinal accuracy using best validation checkpoint:")
print(f"  Train: {train_acc:.2%}")
print(f"  Val:   {val_acc:.2%}")
print(f"  Test:  {test_acc:.2%}")

# ==============================
# SAVE CHECKPOINT FOR LIVE DEMO
# ==============================

checkpoint_path = os.path.join(OUTPUT_MODEL, "mudra_mlp_frame_level.pt")
torch.save(
    {
        "model_state_dict": best_state,
        "input_dim": input_dim,
        "class_names": list(class_names),
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
    },
    checkpoint_path,
)
print(f"Saved: {checkpoint_path}")

# ==============================
# CLASSIFICATION REPORT
# ==============================

print("\nClassification report:")
print(classification_report(
    all_labels,
    all_preds,
    target_names=class_names,
    digits=3,
    zero_division=0
))

# ==============================
# CONFUSION MATRIX
# ==============================

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(cm, display_labels=class_names)

fig, ax = plt.subplots(figsize=(10, 10))
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("Confusion matrix — frame-level test set", fontsize=13)
plt.xticks(rotation=45, ha="right", fontsize=9)
plt.yticks(fontsize=9)
plt.tight_layout()

cm_path = os.path.join(RESULTS_PATH, "mlp_confusion_matrix_frame_level.png")
plt.savefig(cm_path, dpi=150)
plt.show()
print(f"Saved: {cm_path}")

# ==============================
# TRAINING CURVES
# ==============================

epochs_ran = range(1, len(train_losses) + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(epochs_ran, train_losses, label="train", linewidth=2)
ax1.plot(epochs_ran, val_losses, label="val", linewidth=2, linestyle="--")
ax1.set_title("Loss per epoch")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Cross-entropy loss")
ax1.legend()
ax1.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

ax2.plot(epochs_ran, [a * 100 for a in train_accs], label="train", linewidth=2)
ax2.plot(epochs_ran, [a * 100 for a in val_accs], label="val", linewidth=2, linestyle="--")
ax2.set_title("Accuracy per epoch")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.set_ylim(0, 105)
ax2.legend()
ax2.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

plt.tight_layout()
curves_path = os.path.join(RESULTS_PATH, "mlp_training_curves_frame_level.png")
plt.savefig(curves_path, dpi=150)
plt.show()
print(f"Saved: {curves_path}")

# ==============================
# PER-CLASS SUMMARY
# ==============================

prec, rec, f1, support = precision_recall_fscore_support(
    all_labels,
    all_preds,
    labels=range(len(class_names)),
    zero_division=0
)

print("\nPer-class breakdown:")
print(f"{'Sign':<10} {'Precision':>10} {'Recall':>10} {'F1':>8} {'Support':>9}")
print("-" * 52)
for i, name in enumerate(class_names):
    flag = " <-- check" if f1[i] < 0.6 else ""
    print(f"{name:<10} {prec[i]:>10.3f} {rec[i]:>10.3f} {f1[i]:>8.3f} {int(support[i]):>9}{flag}")
print("-" * 52)
print(f"{'macro avg':<10} {prec.mean():>10.3f} {rec.mean():>10.3f} {f1.mean():>8.3f}")
