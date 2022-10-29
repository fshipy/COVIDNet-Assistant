from sklearn.metrics import roc_auc_score
from tensorflow.keras.callbacks import Callback
import tensorflow.keras as tk
import tensorflow.keras.layers as layers

L2RegFunc = tk.regularizers.l2
model_config = {
    "verified_only": {
        "cnn": {"dropout": 0.0},
        "resnet": {
            0: {
                "trimmed": True,
                "dropout": 0.01,
                "conv_stride": 2,
                "kernels": [3, 3, 3],
            },
            1: {
                "trimmed": True,
                "dropout": 0.275,
                "conv_stride": 2,
                "kernels": [3, 3, 3, 3, 3],
            },
            2: {
                "trimmed": True,
                "dropout": 0.275,
                "conv_stride": 2,
                "kernels": [7, 5, 5, 3, 3],
            },
        },
        "mobilenet": {
            0: {
                "depth_multiplier": 1,
                "dropout": 0.04,
                "l2_reg": 0.015,
                "conv_stride": 2,
                "kernels": [3, 3, 3],
                "depth_conv_strides": [2],
                "extra_block": True,
            },
            1: {
                "depth_multiplier": 1,
                "dropout": 0.04,
                "l2_reg": 0.015,
                "conv_stride": 1,
                "kernels": [3, 3, 3, 3, 3, 3, 3],
                "depth_conv_strides": [2, 2, 2],
                "extra_block": False,
            },
            2: {
                "depth_multiplier": 1,
                "dropout": 0.6,
                "l2_reg": 0.02,
                "conv_stride": 1,
                "kernels": [9, 9, 9, 7, 7, 3, 3],
                "depth_conv_strides": [2, 2, 2],
                "extra_block": False,
            },
        },
    },
    "all": {
        "cnn": {"dropout": 0.1},
        "resnet": {
            0: {
                "trimmed": True,
                "dropout": 0.01,
                "conv_stride": 2,
                "kernels": [3, 3, 3],
            },
            1: {
                "trimmed": True,
                "dropout": 0.42,
                "conv_stride": 2,
                "kernels": [3, 3, 3, 3, 3],
            },
            2: {
                "trimmed": True,
                "dropout": 0.45,
                "conv_stride": 2,
                "kernels": [7, 5, 5, 3, 3],
            },
        },
        "mobilenet": {
            0: {
                "depth_multiplier": 1,
                "dropout": 0.04,
                "l2_reg": 0.015,
                "conv_stride": 2,
                "kernels": [3, 3, 3],
                "depth_conv_strides": [2],
                "extra_block": True,
            },
            1: {
                "depth_multiplier": 1,
                "dropout": 0.06,
                "l2_reg": 0.0,
                "conv_stride": 1,
                "kernels": [3, 3, 3, 3, 3, 3, 3],
                "depth_conv_strides": [2, 2, 2],
                "extra_block": False,
            },
            2: {
                "depth_multiplier": 1,
                "dropout": 0.10,
                "l2_reg": 0.01,
                "conv_stride": 1,
                "kernels": [9, 9, 9, 7, 7, 3, 3],
                "depth_conv_strides": [2, 2, 2],
                "extra_block": False,
            },
        },
    },
}


def classifier(x, n_class=2, dropout=0.0, l2_reg=0.0):
    l2_reg = L2RegFunc(l2_reg) if l2_reg > 0 else None
    out_activation = "softmax" if n_class >= 2 else "sigmoid"
    x = layers.GlobalAveragePooling2D()(x)
    if dropout > 0.0:
        x = layers.Dropout(dropout)(x)
    x = layers.Dense(n_class, kernel_regularizer=l2_reg)(x)
    outp = layers.Activation(out_activation, name="output")(x)
    return outp


class AUCCallback(Callback):
    def __init__(self, x, y, auc_name):
        self.x = x
        self.y = y
        self.auc_name = auc_name
        self.history = []

    def max_auc(self):
        max_val = max(self.history)
        max_idx = self.history.index(max_val)
        return (max_val, max_idx)

    def get_roc_history(self):
        return self.history

    def on_epoch_end(self, epoch, logs={}):
        y_preds = self.model.predict(self.x)
        auc = roc_auc_score(self.y, y_preds)
        self.history.append(auc)
        logs[self.auc_name] = auc
        return

    def on_test_end(self, batch, logs={}):
        if self.auc_name == "test_auc":
            y_preds = self.model.predict(self.x)
            auc = roc_auc_score(self.y, y_preds)
            self.history.append(auc)
            logs[self.auc_name] = auc
        return
