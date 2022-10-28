import argparse
from dataclasses import dataclass, fields
from pathlib import Path

import tensorflow.keras as keras
from tqdm.keras import TqdmCallback

from . import models
from data import CovidCoughDataset
import os
import tensorflow as tf
from utility.train_utils import *
import numpy as np
import matplotlib.pyplot as plt

@dataclass
class TrainConfig:
    folds: int
    epochs: int
    initial_learning_rate: float
    batch_size: int

    dropout: float = 0.01
    l2_reg: float = 0.015
    trimmed: bool = False
    arch_index: int = None
    balance_class: bool = False
    verified_only: bool = False
    binary_class: bool = False
    lr_scheduler: str = None
    lr_decay_steps: int = 10
    lr_decay_rate: float = 0.95
    optimizer: str = "Adam"
    cnn_extra_block: int = 0
    reduce_lr_plateau: bool = False
    reduce_lr_plateau_factor: float = 0.75
    reduce_lr_plateau_monitor: str = "val_loss"
    reduce_lr_plateau_patience: int = 1
    override_model_params: bool = False
    early_stopping: bool = False
    early_stopping_patience: int = 0


def train(model_name, train_dataset, val_dataset, model_dir, train_plot_cache, config):
    for fold in range(config.folds):
        model = choose_model(model_name, train_dataset, config)
        train_dataset.set_fold(fold, config.folds)
        train_y = np.argmax(train_dataset.labels, -1) if config.binary_class else train_dataset.labels
        val_y = np.argmax(val_dataset.labels, -1) if config.binary_class else val_dataset.labels
        val_auc = models.AUCCallback(
            val_dataset.data, val_y, auc_name="my_val_auc"
        )
        train_auc = models.AUCCallback(
            train_dataset.data, train_y, auc_name="my_train_auc"
        )
        model_ckpt_callback = keras.callbacks.ModelCheckpoint(
            filepath=str((model_dir / f"{model_name}-{fold}.h5").resolve()),
            save_weights_only=False,
            monitor="my_val_auc",
            mode="max",
            save_best_only=True,
            verbose=0,
        )
        
        lr_schedule = None
        if config.lr_scheduler == "PiecewiseConstantDecay":
            boundaries = [val * len(train_dataset) for val in [0, 10, 20, 30, 40, 75]]
            values = [
                config.initial_learning_rate / val for val in [1, 2, 4, 8, 16, 32, 64]
            ]
            lr_schedule = keras.optimizers.schedules.PiecewiseConstantDecay(
                boundaries, values
            )
        elif config.lr_scheduler == "ExponentialDecay":
            lr_schedule = keras.optimizers.schedules.ExponentialDecay(
                config.initial_learning_rate, config.lr_decay_steps, config.lr_decay_rate
            )
        elif config.lr_scheduler == "CosineDecay":
            lr_schedule = CosineDecay(
                config.initial_learning_rate, config.lr_decay_steps
            )
        if lr_schedule:
            optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
        else:
            optimizer = keras.optimizers.Adam(learning_rate=config.initial_learning_rate)
        loss = keras.losses.BinaryCrossentropy() if config.binary_class else keras.losses.CategoricalCrossentropy()
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=[
                "accuracy",
                keras.metrics.AUC(name="auc"),
                keras.metrics.Recall(name="recall"),
                keras.metrics.Precision(name="precision"),
                keras.metrics.TrueNegatives(name="tn"),
                keras.metrics.TruePositives(name="tp"),
                keras.metrics.FalseNegatives(name="fn"),
                keras.metrics.FalsePositives(name="fp"),
            ],
        )
        class_weight = None
        if config.balance_class:
            n_neg = sum(train_dataset.labels[:, 0] == 0)
            n_pos = sum(train_dataset.labels[:, 0] == 1)
            n_total = len(train_dataset.labels)
            class_weight = {
                0 : (1 / n_neg) * (n_total / 2.0),
                1 : (1 / n_pos) * (n_total / 2.0)
            }
            print("class_weight:", class_weight)
            print(n_neg, n_pos, n_total)
        
        callbacks=[
                    train_auc,
                    val_auc,
                    model_ckpt_callback,
                    TqdmCallback(epochs=config.epochs, verbose=0),  
        ]
        if config.early_stopping:
            callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=config.early_stopping_patience))
        if config.reduce_lr_plateau:
            callbacks.append(
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor=config.reduce_lr_plateau_monitor,
                    factor=config.reduce_lr_plateau_factor,
                    patience=config.reduce_lr_plateau_patience,
                    verbose=1,
                )
            )
        if config.binary_class:
            history = model.fit(
                x=train_dataset.data,
                y=np.argmax(train_dataset.labels, -1),
                validation_data=(val_dataset.data,np.argmax(val_dataset.labels, -1)),
                batch_size=train_dataset.batch_size,
                epochs=config.epochs,
                shuffle=True,
                verbose=0,
                callbacks=callbacks,
                class_weight=class_weight,
            )
        else:
            history = model.fit(
                x=train_dataset,
                validation_data=val_dataset,
                epochs=config.epochs,
                verbose=0,
                callbacks=callbacks,
                class_weight=class_weight,
            )
        plot_metrics(history, model_name, os.path.join(train_plot_cache, f"train_log_{model_name}.png"))

def plot_metrics(history, model_name, save_fig="train_log.png"):
    plt.figure(figsize=(16, 16))
    metrics = ["loss", 'auc', 'recall', 'precision', 'acc', "tn", 'tp', 'fp', 'fn']
    for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        plt.subplot(3,3,n+1)
        plt.plot(history.epoch, history.history[metric], label='Train')
        plt.plot(history.epoch, history.history['val_'+metric],
                 linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)

        plt.legend()
    plt.suptitle("Training " + model_name + " no balance", fontsize=14)
    plt.savefig(save_fig)
    plt.close()
    

def choose_model(model_name, train_dataset, config):
    inp = keras.Input(shape=train_dataset.data.shape[1:], name="input")
    model_name = model_name.lower()
    n_class = 1 if config.binary_class else 2 
    if model_name.startswith("cnn"):
        return models.cnn_model(inp, dropout=config.dropout, n_class=n_class, n_extra_block=config.cnn_extra_block)
    elif model_name.startswith("resnet18"):
        return models.resnet18(inp, dropout=config.dropout, n_class=n_class)
    elif model_name.startswith("resnet"):
        return models.resnet(
            inp,
            dropout=config.dropout,
            trimmed=config.trimmed,
            arch_index=config.arch_index,
            verified_only=config.verified_only,
            n_class=n_class,
            override_param=config.override_model_params,
        )
    elif model_name.startswith("mobilenet"):
        return models.mobileNetV1(
            inp,
            dropout=config.dropout,
            l2_reg=config.l2_reg,
            arch_index=config.arch_index,
            verified_only=config.verified_only,
            n_class=n_class,
            override_param=config.override_model_params,
        )
    elif model_name.startswith("full_mobilenet"):
        return models.full_mobileNet_s(
            inp,
            dropout=config.dropout,
            l2_reg=config.l2_reg,
            n_class=n_class,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Covid-Cough models.")
    parser.add_argument(
        "--train_data",
        type=Path,
        required=True,
        help="Path to training data. It should be a pickle file.",
    )
    parser.add_argument(
        "--val_data",
        type=Path,
        required=True,
        help="Path to validation data. It should be a pickle file.",
    )
    parser.add_argument(
        "--model_dir",
        type=Path,
        required=True,
        help="Path to output model directory. The model checkpoints will be saved there.",
    )
    model_names = ["cnn", "resnet18", "resnet", "mobilenet", "full_mobilenet"]
    parser.add_argument(
        "--model_name",
        required=True,
        #choices=model_names,
        help=f"Model name. Choose one from {model_names}.",
    )

    # training config
    parser.add_argument("--folds", type=int, default=1, help="Number of folds to use.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs.")
    parser.add_argument(
        "--initial_learning_rate",
        type=float,
        default=0.0002,
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.01,
        help="Drop out probablity to use. Will be applied to all models.",
    )
    parser.add_argument(
        "--l2_reg",
        type=float,
        default=0.015,
        help="l2 regularization to use. Will be applied to mobilenet.",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument(
        "--arch_index",
        type=int,
        default=None,
        help="""Index of preconfigured architecture parameters,
                see covid_cough/models/utils.py for details""",
    )
    parser.add_argument(
        "--verified_only", action="store_true", help="Only use verified data"
    )
    parser.add_argument(
        "--balance_class", action="store_true", help="Train with balanced class"
    )
    parser.add_argument(
        "--binary_class", action="store_true", help="Train with binary class mode"
    )
    parser.add_argument(
        "--train_plot_cache",
        type=Path,
        default=Path("train_plots"),
        help="Path to save the training plots.",
    )
    parser.add_argument(
        "--gpu_index",
        type=int,
        default=0,
        help="which gpu to use",
    )
    parser.add_argument(
        "--cnn_extra_block",
        type=int,
        default=0,
        help="extra cnn block for CNN model",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default=None,
        help="learning rate scheduler",
    )
    parser.add_argument(
        "--lr_decay_steps",
        type=int,
        default=10,
        help="learning rate scheduler decay step if necessary",
    )
    parser.add_argument(
        "--lr_decay_rate",
        type=float,
        default=0.95,
        help="learning rate scheduler decay rate if necessary",
    )
    parser.add_argument(
        "--override_model_params", action="store_true", help="use user defined config to override some pre-defined model config"
    )
    parser.add_argument(
        "--early_stopping", action="store_true", help="Train with early stopping"
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=10,
        help="early stopping patience",
    )
    parser.add_argument(
        "--reduce_lr_plateau", action="store_true", help="use ReduceLROnPlateau"
    )
    parser.add_argument(
        "--reduce_lr_plateau_factor",
        type=float,
        default=0.75,
        help="ReduceLROnPlateau factor",
    )
    parser.add_argument(
        "--reduce_lr_plateau_monitor",
        type=str,
        default="val_loss",
        help="ReduceLROnPlateau monitor",
    )
    parser.add_argument(
        "--reduce_lr_plateau_patience",
        type=int,
        default=2,
        help="ReduceLROnPlateau patience",
    )



    args = parser.parse_args()

    if not os.path.exists(args.train_plot_cache):
        os.makedirs(args.train_plot_cache)


    train_config = TrainConfig(
        **{
            k.name: getattr(args, k.name)
            for k in fields(TrainConfig)
            if k.name in vars(args)
        }
    )

    train_dataset = CovidCoughDataset(args.train_data, train_config.batch_size)
    val_dataset = CovidCoughDataset(args.val_data, train_config.batch_size)

    args.model_dir.mkdir(parents=True, exist_ok=True)
    with tf.device(f'/device:GPU:{args.gpu_index}'):
        train(args.model_name, train_dataset, val_dataset, args.model_dir, args.train_plot_cache, train_config)
