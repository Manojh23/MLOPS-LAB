"""
Train a TensorFlow (Keras) model on the Iris dataset.

Model:
- Dense neural network classifier (softmax over 3 classes)

Run:
  python src/train_model.py
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")

TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")

FEATURE_COLS = ["sepal_length_cm", "sepal_width_cm", "petal_length_cm", "petal_width_cm"]
LABEL_COL = "label"


def load_xy(csv_path: str):
    df = pd.read_csv(csv_path)
    x = df[FEATURE_COLS].astype("float32").to_numpy()
    y = df[LABEL_COL].astype("int32").to_numpy()
    return x, y


def main() -> None:
    print("=== TensorFlow (Keras) Iris Classifier ===")
    x_train, y_train = load_xy(TRAIN_CSV)
    x_test, y_test = load_xy(TEST_CSV)

    # Split train into train/val for monitoring
    x_tr, x_val, y_tr, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    # Normalize features (simple z-score using train split stats)
    mean = x_tr.mean(axis=0, keepdims=True)
    std = x_tr.std(axis=0, keepdims=True) + 1e-7
    x_tr_n = (x_tr - mean) / std
    x_val_n = (x_val - mean) / std
    x_test_n = (x_test - mean) / std

    num_classes = 3

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(len(FEATURE_COLS),)),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    print("\nModel summary:")
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=15, restore_best_weights=True)
    ]

    history = model.fit(
        x_tr_n, y_tr,
        validation_data=(x_val_n, y_val),
        epochs=200,
        batch_size=16,
        verbose=1,
        callbacks=callbacks,
    )

    test_loss, test_acc = model.evaluate(x_test_n, y_test, verbose=0)
    print(f"\nTest accuracy: {test_acc:.4f} | Test loss: {test_loss:.4f}")

    # Save model
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_model")
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "iris_dense_classifier")
    model.save(model_path)
    print(f"Saved model to: {model_path}")

    # Quick demo prediction
    sample = x_test_n[:3]
    probs = model.predict(sample, verbose=0)
    preds = probs.argmax(axis=1)
    print("\nSample predictions (first 3 test rows):")
    for i, (p, y) in enumerate(zip(preds, y_test[:3])):
        print(f"  row {i}: pred={int(p)} true={int(y)} probs={np.round(probs[i], 3)}")


if __name__ == "__main__":
    main()
