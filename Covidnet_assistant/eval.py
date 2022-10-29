import argparse
import tensorflow as tf
from pathlib import Path
import numpy as np
import tensorflow.keras as keras
from sklearn.metrics import (
    roc_auc_score,
    recall_score,
    precision_score,
    f1_score,
    accuracy_score,
    average_precision_score,
)

from data import CovidCoughDataset


def eval(model_path, test_dataset, binary_class=False, threshold=0.5):
    model = keras.models.load_model(model_path)
    if binary_class:
        test_dataset.labels = np.argmax(test_dataset.labels, -1)
    predictions_conf = model.predict(test_dataset.data)
    predictions = predictions_conf >= threshold
    auc = roc_auc_score(test_dataset.labels, predictions_conf)
    recall = recall_score(test_dataset.labels, predictions)
    precision = precision_score(test_dataset.labels, predictions)
    f1 = f1_score(test_dataset.labels, predictions)
    accuracy = accuracy_score(test_dataset.labels, predictions)
    ap_score = average_precision_score(test_dataset.labels, predictions_conf)

    return auc, recall, precision, f1, accuracy, ap_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate covid-cough models.")
    parser.add_argument("--model_path", type=Path, required=True, help="Path to model.")
    parser.add_argument(
        "--test_data",
        type=Path,
        required=True,
        help="Path to test data in pickle format",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument(
        "--binary_class", action="store_true", help="Train with binary class mode"
    )
    args = parser.parse_args()
    test_dataset = CovidCoughDataset(args.test_data, args.batch_size)
    with tf.device("/device:GPU:7"):
        auc, recall, precision, f1, accuracy, ap_score = eval(
            args.model_path, test_dataset, args.binary_class
        )
    print("AUC score: ", auc)
    print("Recall score: ", recall)
    print("Precision score: ", precision)
    print("F1 score: ", f1)
    print("Accuracy score: ", accuracy)
    print("Average Precision score: ", ap_score)
