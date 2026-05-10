import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==============================
# CONFIG
# ==============================

INPUT_FOLDER = "2_features"
OUTPUT_FOLDER = "4_feature_data_info"
PLOT_FOLDER = os.path.join(OUTPUT_FOLDER, "plots")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(PLOT_FOLDER, exist_ok=True)

# ==============================
# ANALYSIS FUNCTION
# ==============================

def analyze_file(filepath):

    data = pd.read_csv(filepath)
    frames = data.values

    left = frames[:, :63]
    right = frames[:, 63:]

    # Detect missing hands
    left_missing = np.all(left == 0, axis=1)
    right_missing = np.all(right == 0, axis=1)

    both_missing = left_missing & right_missing

    stats = {
        "total_frames": len(frames),
        "left_missing_%": np.mean(left_missing) * 100,
        "right_missing_%": np.mean(right_missing) * 100,
        "both_missing_%": np.mean(both_missing) * 100
    }

    return stats, left_missing, right_missing


# ==============================
# MAIN LOOP
# ==============================

results = []

for file in os.listdir(INPUT_FOLDER):

    if not file.endswith(".csv"):
        continue

    filepath = os.path.join(INPUT_FOLDER, file)
    class_name = file.replace(".csv", "")

    stats, left_missing, right_missing = analyze_file(filepath)

    stats["class"] = class_name
    results.append(stats)

    # ==============================
    # PLOT 1: Missing over time
    # ==============================

    plt.figure()
    plt.plot(left_missing.astype(int), label="Left Missing")
    plt.plot(right_missing.astype(int), label="Right Missing")
    plt.title(f"Missing Hands Timeline - {class_name}")
    plt.xlabel("Frame")
    plt.ylabel("Missing (1 = yes)")
    plt.legend()
    plt.savefig(os.path.join(PLOT_FOLDER, f"{class_name}_timeline.png"))
    plt.close()

    # ==============================
    # PLOT 2: Bar chart
    # ==============================

    plt.figure()
    plt.bar(
        ["Left", "Right"],
        [stats["left_missing_%"], stats["right_missing_%"]]
    )
    plt.title(f"Missing Hand % - {class_name}")
    plt.ylabel("Percentage")
    plt.savefig(os.path.join(PLOT_FOLDER, f"{class_name}_bar.png"))
    plt.close()


# ==============================
# SAVE SUMMARY
# ==============================

df = pd.DataFrame(results)
df = df.sort_values("class")

df.to_csv(os.path.join(OUTPUT_FOLDER, "summary.csv"), index=False)

print("Analysis complete!")
print(df)