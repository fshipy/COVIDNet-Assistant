import tensorflow.keras as tk
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model

from .utils import model_config, classifier

L2RegFunc = tk.regularizers.l2


def _conv_block(x, n_filters, kernel=(3, 3), strides=(1, 1), dropout=0.0, l2_reg=0.0):
    """
    initial conv block
    """
    fx = layers.Conv2D(
        n_filters,
        kernel,
        padding="same",
        use_bias=False,
        strides=strides,
        kernel_regularizer=L2RegFunc(l2_reg),
    )(x)

    if dropout:
        fx = layers.SpatialDropout2D(dropout)(fx)

    fx = layers.BatchNormalization()(fx)
    return layers.ReLU(6.0)(fx)


def _depthwise_conv_block(
    x,
    pointwise_conv_filters,
    depth_multiplier=1,
    depthwise_kernel_size=(3, 3),
    strides=(1, 1),
    dropout1=0.0,  # after deepwise conv
    dropout2=0.0,
    l2_reg=0.0,
):  # after pointwise conv

    if strides == (1, 1):
        x = x
    else:
        x = layers.ZeroPadding2D(((0, 1), (0, 1)))(x)

    x = layers.DepthwiseConv2D(
        depthwise_kernel_size,
        padding="same" if strides == (1, 1) else "valid",
        depth_multiplier=depth_multiplier,
        strides=strides,
        kernel_regularizer=L2RegFunc(l2_reg),
        use_bias=False,
    )(x)
    if dropout1:
        x = layers.SpatialDropout2D(dropout1)(x)

    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.0)(x)

    x = layers.Conv2D(
        pointwise_conv_filters,
        (1, 1),
        padding="same",
        use_bias=False,
        strides=(1, 1),
        kernel_regularizer=L2RegFunc(l2_reg),
    )(x)

    if dropout2:
        x = layers.SpatialDropout2D(dropout2)(x)

    x = layers.BatchNormalization()(x)
    return layers.ReLU(6.0)(x)


def mobileNetV1(
    inp,
    depth_multiplier=1,
    dropout=0.05,
    l2_reg=0.00,
    conv_stride=1,
    kernels=[9, 9, 9, 7, 7, 3, 3],
    depth_conv_strides=[2, 2, 2],
    arch_index=None,
    verified_only=False,
    n_class=2,
    override_param=False,
    extra_block=False,
):
    """This is a custom mobileNetV1 with dropout, original paper has 13 depthwise_conv_block"""
    # if use preset architecture, override user model configs
    if arch_index is not None:
        t = "verified_only" if verified_only else "all"
        depth_multiplier = model_config[t]["mobilenet"][arch_index]["depth_multiplier"]
        if not override_param:
            dropout = model_config[t]["mobilenet"][arch_index]["dropout"]
            l2_reg = model_config[t]["mobilenet"][arch_index]["l2_reg"]
        conv_stride = model_config[t]["mobilenet"][arch_index]["conv_stride"]
        kernels = model_config[t]["mobilenet"][arch_index]["kernels"]
        depth_conv_strides = model_config[t]["mobilenet"][arch_index][
            "depth_conv_strides"
        ]
        extra_block = model_config[t]["mobilenet"][arch_index]["extra_block"]

    x = _conv_block(
        inp,
        32,
        kernel=(kernels[0], kernels[0]),
        strides=(conv_stride, conv_stride),
        l2_reg=l2_reg,
    )

    block_pair = 0
    out_channel = 64
    n_pairs = (len(kernels) - 1) // 2
    while block_pair < n_pairs:
        x = _depthwise_conv_block(
            x,
            out_channel,
            depth_multiplier,
            depthwise_kernel_size=(
                kernels[block_pair * 2 + 1],
                kernels[block_pair * 2 + 1],
            ),
            strides=(depth_conv_strides[block_pair], depth_conv_strides[block_pair]),
            dropout2=dropout,
            l2_reg=l2_reg,
        )

        x = _depthwise_conv_block(
            x,
            out_channel,
            depth_multiplier,
            depthwise_kernel_size=(
                kernels[block_pair * 2 + 2],
                kernels[block_pair * 2 + 2],
            ),
            l2_reg=l2_reg,
        )
        out_channel *= 2
        block_pair += 1

    if extra_block:
        x = _depthwise_conv_block(x, out_channel, depth_multiplier, l2_reg=l2_reg)

    outp = classifier(x, dropout=dropout, l2_reg=l2_reg, n_class=n_class)
    model = Model(inp, outp)
    return model


def full_mobileNet_s(
    inp, alpha=1, dropout=0.001, l2_reg=0.0, depth_multiplier=1, n_class=2
):

    x = _conv_block(inp, 32, kernel=(3, 3), strides=(2, 2), l2_reg=l2_reg)
    x = _depthwise_conv_block(
        x, 64, depth_multiplier, depthwise_kernel_size=(3, 3), l2_reg=l2_reg
    )
    x = _depthwise_conv_block(
        x,
        128,
        depth_multiplier,
        depthwise_kernel_size=(3, 3),
        strides=(2, 2),
        dropout2=dropout,
        l2_reg=l2_reg,
    )
    x = _depthwise_conv_block(
        x, 128, depth_multiplier, depthwise_kernel_size=(3, 3), l2_reg=l2_reg
    )

    x = _depthwise_conv_block(
        x,
        256,
        depth_multiplier,
        depthwise_kernel_size=(3, 3),
        strides=(2, 2),
        dropout2=dropout,
        l2_reg=l2_reg,
    )
    x = _depthwise_conv_block(
        x, 256, depth_multiplier, depthwise_kernel_size=(3, 3), l2_reg=l2_reg
    )

    x = _depthwise_conv_block(
        x,
        512,
        depth_multiplier,
        depthwise_kernel_size=(3, 3),
        strides=(2, 2),
        dropout2=dropout,
        l2_reg=l2_reg,
    )
    x = _depthwise_conv_block(
        x, 512, depth_multiplier, depthwise_kernel_size=(3, 3), l2_reg=l2_reg
    )
    x = _depthwise_conv_block(
        x, 512, depth_multiplier, depthwise_kernel_size=(3, 3), l2_reg=l2_reg
    )
    x = _depthwise_conv_block(
        x, 512, depth_multiplier, depthwise_kernel_size=(3, 3), l2_reg=l2_reg
    )
    x = _depthwise_conv_block(
        x, 512, depth_multiplier, depthwise_kernel_size=(3, 3), l2_reg=l2_reg
    )
    x = _depthwise_conv_block(
        x, 512, depth_multiplier, depthwise_kernel_size=(3, 3), l2_reg=l2_reg
    )
    x = _depthwise_conv_block(
        x,
        1024,
        depth_multiplier,
        depthwise_kernel_size=(3, 3),
        strides=(2, 2),
        dropout2=dropout,
        l2_reg=l2_reg,
    )
    x = _depthwise_conv_block(
        x,
        1024,
        depth_multiplier,
        depthwise_kernel_size=(3, 3),
        dropout2=dropout,
        l2_reg=l2_reg,
    )

    outp = classifier(x, dropout=dropout, l2_reg=l2_reg, n_class=n_class)
    model = Model(inp, outp)
    return model
