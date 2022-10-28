import time
import os
import numpy as np
import argparse
from pathlib import Path
from scipy import stats
import json
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"]=""
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data', type=Path, required=False)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--latency_only', action='store_true')
    parser.add_argument('--models_parent', action='store_true', help="test a bunch of models under model")
    parser.add_argument(
        "--binary_class", action="store_true", help="Train with binary class mode"
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="Path to model (tf graph) checkpoint.",
        required=True,
    )
    return parser.parse_args()

def inference_latency(sess, binary_class, input_shape=(1, 32, 328, 1), n=1000, warm_up=20, trimmed_prob=0.1):
    dummy_input = np.ones(input_shape)
    latency = []
    output_tensor = "output/Softmax:0" if not binary_class else "output/Sigmoid:0"
    for _ in range(warm_up):
        sess.run(output_tensor, feed_dict={"input:0": dummy_input})
    
    for _ in range(n):
        start_time = time.time()
        sess.run(output_tensor, feed_dict={"input:0": dummy_input})
        end_time = time.time()
        latency.append(end_time - start_time)
    
    latency = np.array(latency)
    strimmed_latency = stats.trimboth(latency, trimmed_prob)
    return np.mean(strimmed_latency), np.std(strimmed_latency)
    #return stats.trim_mean(latency, trimmed_prob)

def test_auc(sess, test_data_path, batch_size, binary_class, threshold=0.5):
    from data import CovidCoughDataset
    from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, recall_score, precision_score, f1_score, average_precision_score, confusion_matrix
    test_dataset = CovidCoughDataset(test_data_path, batch_size)
    X_test = test_dataset.data
    y_test = test_dataset.labels
    if binary_class:
        y_test = np.argmax(y_test, -1)
    output_tensor = "output/Softmax:0" if not binary_class else "output/Sigmoid:0"

    y_pred_conf = sess.run(output_tensor, feed_dict={"input:0": X_test})

    
    if not binary_class:
        y_test = np.argmax(y_test, axis=1)
        y_pred = np.argmax(y_pred_conf, axis=1)
    else:
        y_pred = y_pred_conf >= threshold
    auc = roc_auc_score(y_test, y_pred_conf)
    accuracy = accuracy_score(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    ap_score = average_precision_score(y_test, y_pred_conf)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn+fp)
    sensitivity = tp / (tp+fn)
    if auc < 0.5:
       print("auc is less than 0.5, y_pred:", y_pred, y_pred.shape)

    return auc, accuracy, balanced_accuracy, recall, precision, f1, ap_score, specificity, sensitivity

def count_params(sess):
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print("total trainable variables", total_parameters)
    return total_parameters

def main(model_path, test_data_path, batch_size, latency_only, binary_class):
    print("model_path:", model_path)

    tf.reset_default_graph()

    with tf.Session() as sess:

        saver = tf.train.import_meta_graph(os.path.join(model_path, 'model.meta'))

        saver.restore(sess, tf.train.latest_checkpoint(model_path))

        total_parameters = count_params(sess)

        writer = tf.summary.FileWriter(os.path.join(model_path, 'graph'), sess.graph)

        out_dict = {}

        latency, latency_std = inference_latency(sess, binary_class)
        out_dict["model_path:"] = model_path
        out_dict["trainable variables"] = total_parameters
        out_dict["latency_mean"] = round(latency * 1000, ndigits = 4)
        out_dict["latency_std"] = round(latency_std * 1000, ndigits = 4)

        print("inference latency mean:", round(latency * 1000, ndigits = 4), "ms")
        print("inference latency std:", round(latency_std * 1000, ndigits = 4), "ms")

        if not latency_only:

            auc, accuracy, balanced_accuracy, recall, precision, f1, ap_score, specificity, sensitivity = test_auc(sess, test_data_path, batch_size, binary_class)
            print("AUC:", auc)
            print("Accuracy:", accuracy)
            print("Balanced Accuracy:", balanced_accuracy)
            print("Recall score: ", recall)
            print("Precision score: ", precision)
            print("F1 score: ", f1)
            print("Average Precision score: ", ap_score)
            print("Specificity score: ", specificity)
            print("Sensitivity score: ", sensitivity)

            out_dict["auc"] = auc
            out_dict["accuracy"] = accuracy
            out_dict["balanced_accuracy"] = balanced_accuracy
            out_dict["recall"] = recall
            out_dict["precision"] = precision
            out_dict["f1"] = f1
            out_dict["ap_score"] = ap_score
            out_dict["specificity"] = specificity
            out_dict["sensitivity"] = sensitivity

        writer.close()
    
    return out_dict

if __name__ == "__main__":
    args = parse_args()
    if args.models_parent:
        result_dict = {}
        for model_dir in os.scandir(args.model):
            if model_dir.is_dir():
                out_dict = main(model_dir.path, args.test_data, args.batch_size, args.latency_only, args.binary_class)
                result_dict[str(model_dir.path)] = out_dict
        
        with open(f'test_result_{str(args.model).replace("/", "_")}.json', 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=4)
    else:
        main(args.model, args.test_data, args.batch_size, args.latency_only, args.binary_class)