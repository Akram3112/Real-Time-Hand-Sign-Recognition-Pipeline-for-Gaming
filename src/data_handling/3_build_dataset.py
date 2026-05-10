import os
import shutil

import numpy as np
import pandas as pd

PROCESSED_PATH = "../../3_features_processed"
SPLIT_PATH = "../../4_dataset_POINT"

LABELS = [
    "bird", "boar", "dog", "dragon", "hare", "horse",
    "monkey", "ox", "ram", "rat", "snake", "tiger"
]

VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42

# Person 6 and its mirrored version stay in the test set, (because person 6 is the one testing on live data)
FORCED_TEST_PERSONS = {"6"}


def base_person_id(person_id) -> str:
    base_id = str(person_id).replace("_m", "")
    return str(int(float(base_id)))


def make_person_split(all_people):
    unique_people = np.array(sorted([str(p) for p in all_people], key=int))

    missing_forced_test = FORCED_TEST_PERSONS - set(unique_people)
    if missing_forced_test:
        raise RuntimeError(
            f"Forced test person(s) not found in data: {sorted(missing_forced_test)}"
        )

    available_for_random_split = [str(p) for p in unique_people if str(p) not in FORCED_TEST_PERSONS]

    rng = np.random.default_rng(SEED)
    shuffled_remaining = np.array(available_for_random_split)
    rng.shuffle(shuffled_remaining)

    num_people = len(unique_people)
    test_count = max(len(FORCED_TEST_PERSONS), int(round(TEST_RATIO * num_people)))
    val_count = max(1, int(round(VAL_RATIO * num_people)))

    extra_test_count = test_count - len(FORCED_TEST_PERSONS)
    extra_test_persons = set(str(p) for p in shuffled_remaining[:extra_test_count])
    test_people = set(FORCED_TEST_PERSONS) | extra_test_persons

    remaining_after_test = [
        str(p) for p in shuffled_remaining[extra_test_count:]
        if str(p) not in test_people
    ]
    val_people = set(remaining_after_test[:val_count])
    train_people = set(remaining_after_test[val_count:])

    if not train_people or not val_people or not test_people:
        raise RuntimeError(
            "Person split produced an empty train/val/test set "
        )

    return train_people, val_people, test_people


def main():
    if not os.path.isdir(PROCESSED_PATH):
        raise FileNotFoundError(f"Processed feature folder not found: {PROCESSED_PATH}")

    data_by_label = {}
    all_people = set()

    for label in LABELS:
        path = os.path.join(PROCESSED_PATH, f"{label}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing processed file for label '{label}': {path}")

        df = pd.read_csv(path)
        if df.empty:
            raise RuntimeError(f"Processed file is empty for label '{label}': {path}")

        df = df.copy()
        df["_base_person_id"] = df["person_id"].apply(base_person_id)
        all_people.update(df["_base_person_id"].unique())
        data_by_label[label] = df

    train_people, val_people, test_people = make_person_split(all_people)

    print("Person-level split:")
    print(f"  Train persons: {sorted(train_people, key=int)}")
    print(f"  Val persons:   {sorted(val_people, key=int)}")
    print(f"  Test persons:  {sorted(test_people, key=int)}")

    if os.path.exists(SPLIT_PATH):
        shutil.rmtree(SPLIT_PATH)

    os.makedirs(SPLIT_PATH, exist_ok=True)

    split_people = {
        "train": train_people,
        "val": val_people,
        "test": test_people,
    }

    summary_rows = []
    split_frames = {split_name: [] for split_name in split_people}

    for label, df in data_by_label.items():
        for split_name, people in split_people.items():
            split_df = df[df["_base_person_id"].isin(people)].drop(columns=["_base_person_id"])
            split_df = split_df.copy()
            split_df["class_id"] = LABELS.index(label)
            # Reorder columns to match original: person_id | class_id | features
            cols = ["person_id", "class_id"] + [
                c for c in split_df.columns if c not in ("person_id", "class_id")
            ]
            split_df = split_df[cols]
            split_frames[split_name].append(split_df)

            summary_rows.append({
                "split": split_name,
                "label": label,
                "rows": len(split_df),
                "people": " ".join(sorted(people, key=int)),
            })

            print(f"Saved {split_name:>5} {label:<7}: {len(split_df)} rows")

    for split_name, frames in split_frames.items():
        combined_df = pd.concat(frames, ignore_index=True)
        # Map split names to match original structure
        if split_name == "val":
            folder_name = "validation"
        else:
            folder_name = split_name
        os.makedirs(os.path.join(SPLIT_PATH, folder_name), exist_ok=True)
        output_path = os.path.join(SPLIT_PATH, folder_name, f"{folder_name}.csv")
        combined_df.to_csv(output_path, index=False)
        print(f"Saved combined {split_name:>5}: {len(combined_df)} rows -> {output_path}")

    summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(SPLIT_PATH, "split_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()