import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model
from .utils import classifier


def cnn_model(inp, dropout=0.0, n_class=2, n_extra_block=0):
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(dropout)(x)

    for i in range(n_extra_block):
        x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
        x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Dropout(dropout)(x)

    x = layers.BatchNormalization()(x)
    outp = classifier(x, n_class=n_class, dropout=dropout)
    model = Model(inp, outp)
    return model
