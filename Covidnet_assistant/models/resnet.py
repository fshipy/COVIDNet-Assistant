import tensorflow.keras.layers as layers
import tensorflow.keras as tk
from tensorflow.keras.models import Model

from .utils import model_config, classifier
L2RegFunc = tk.regularizers.l2

def res_block(x, n_filters, k_size=3, downsample=False, dropout=0.0):
    fx = layers.Conv2D(
        n_filters,
        k_size,
        activation="relu",
        padding="same",
        strides=(1 if not downsample else 2),
    )(x)

    fx = layers.BatchNormalization()(fx)
    fx = layers.Conv2D(n_filters, k_size, padding="same")(fx)
    if dropout:
        fx = layers.SpatialDropout2D(dropout)(fx)

    if downsample:
        x = layers.Conv2D(n_filters, 1, strides=2, padding="same")(x)

    out = layers.Add()([fx, x])
    out = layers.ReLU()(out)
    out = layers.BatchNormalization()(out)
    return out


def resnet18(inp, dropout=0.2, n_class=2):
    """Custom Resnet 18 with dropout."""
    x = layers.Conv2D(64, 7, strides=(2, 2), padding="same")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = res_block(x, 64, downsample=True, dropout=dropout)
    x = res_block(x, 64)

    x = res_block(x, 128, downsample=True, dropout=dropout)
    x = res_block(x, 128)

    x = res_block(x, 256, downsample=True, dropout=dropout)
    x = res_block(x, 256)

    x = res_block(x, 512, downsample=True, dropout=dropout)
    x = res_block(x, 512)

    outp = classifier(x, n_class=n_class, dropout=dropout)
    model = Model(inp, outp)
    return model


def resnet(
    inp,
    dropout=0.2,
    trimmed=False,
    conv_stride=1,
    kernels=[9, 9, 9, 7, 7],
    arch_index=None,
    verified_only=False,
    n_class=2,
    override_param=False,
):
    if (
        arch_index is not None
    ):  # if use preset architecture, override user model configs, unless override_param is true
        t = "verified_only" if verified_only else "all"
        if not override_param:
            dropout = model_config[t]["resnet"][arch_index]["dropout"]
        trimmed = model_config[t]["resnet"][arch_index]["trimmed"]
        conv_stride = model_config[t]["resnet"][arch_index]["conv_stride"]
        kernels = model_config[t]["resnet"][arch_index]["kernels"]

    if trimmed:  # smaller resnet
        x = layers.Conv2D(
            32, kernels[0], strides=(conv_stride, conv_stride), padding="same"
        )(inp)
        x = res_block(x, 32 * 2, k_size=kernels[1], downsample=True, dropout=dropout)
        x = res_block(x, 32 * 2, k_size=kernels[2])
        if len(kernels) > 4:
            x = res_block(x, 64 * 2, k_size=kernels[3], downsample=True, dropout=dropout)
            x = res_block(x, 64 * 2, k_size=kernels[4])
    else:
        x = layers.Conv2D(
            32, kernels[0], strides=(conv_stride, conv_stride), padding="same"
        )(inp)
        x = res_block(x, 32 * 2, k_size=kernels[1], downsample=True, dropout=dropout)
        x = res_block(x, 32 * 2, k_size=kernels[2])
        x = res_block(x, 32 * 2, k_size=kernels[3])
        x = res_block(x, 32 * 2, k_size=kernels[4])

        x = res_block(x, 64 * 2, downsample=True, dropout=dropout)
        x = res_block(x, 64 * 2)
        x = res_block(x, 64 * 2)
        x = res_block(x, 64 * 2)

    x = res_block(x, 128 * 2, k_size=1, downsample=True, dropout=dropout)
    x = res_block(x, 128 * 2, k_size=1)
    outp = classifier(x, n_class=n_class, dropout=dropout)
    model = Model(inp, outp)
    return model
