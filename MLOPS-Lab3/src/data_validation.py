"""
Data Validation with TensorFlow Data Validation (TFDV) on Iris CSV.

What this script does:
1) Loads train/test CSV files.
2) Computes training statistics.
3) Infers a schema from training data.
4) Checks test data for anomalies against the schema.
5) (Optional) Computes drift comparison between train and test stats.

Run:
  python src/data_validation.py
"""

import os
import tensorflow_data_validation as tfdv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")

TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")


def main() -> None:
    print("=== TFDV Iris Data Validation ===")
    print(f"Train CSV: {TRAIN_CSV}")
    print(f"Test  CSV: {TEST_CSV}")

    # 1) Statistics
    train_stats = tfdv.generate_statistics_from_csv(TRAIN_CSV)
    print("\n[1] Generated training statistics.")

    # 2) Schema inference
    schema = tfdv.infer_schema(train_stats)
    print("\n[2] Inferred schema from training stats:")
    tfdv.display_schema(schema)

    # 3) Test statistics + anomalies
    test_stats = tfdv.generate_statistics_from_csv(TEST_CSV)
    print("\n[3] Generated test statistics.")

    anomalies = tfdv.validate_statistics(test_stats, schema)
    print("\n[4] Anomalies in test data (if empty, validation passed):")
    tfdv.display_anomalies(anomalies)

    # 4) Compare train vs test stats (drift-style view)
    print("\n[5] Train vs Test stats comparison (useful for drift checks):")
    tfdv.visualize_statistics(lhs_statistics=train_stats, rhs_statistics=test_stats, lhs_name="train", rhs_name="test")

    print("\nDone.")


if __name__ == "__main__":
    main()
