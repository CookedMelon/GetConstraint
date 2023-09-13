"/home/cc/Workspace/tfconstraint/keras/applications/nasnet.py"
@keras_export(
    "keras.applications.nasnet.NASNetLarge", "keras.applications.NASNetLarge"
)
def NASNetLarge(
    input_shape=None,
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
):
    """Instantiates a NASNet model in ImageNet mode.
    Reference:
    - [Learning Transferable Architectures for Scalable Image Recognition](
        https://arxiv.org/abs/1707.07012) (CVPR 2018)
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.
    Note: each Keras Application expects a specific kind of input preprocessing.
    For NASNet, call `tf.keras.applications.nasnet.preprocess_input` on your
    inputs before passing them to the model.
    Args:
        input_shape: Optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(331, 331, 3)` for NASNetLarge.
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 32.
            E.g. `(224, 224, 3)` would be one valid value.
        include_top: Whether to include the fully-connected
            layer at the top of the network.
        weights: `None` (random initialization) or
            `imagenet` (ImageNet weights).  For loading `imagenet` weights,
            `input_shape` should be (331, 331, 3)
        input_tensor: Optional Keras tensor (i.e. output of
            `layers.Input()`)
            to use as image input for the model.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model
                will be the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a
                2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: Optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        classifier_activation: A `str` or callable. The activation function to
            use on the "top" layer. Ignored unless `include_top=True`. Set
            `classifier_activation=None` to return the logits of the "top"
            layer.  When loading pretrained weights, `classifier_activation` can
            only be `None` or `"softmax"`.
    Returns:
        A Keras model instance.
    Raises:
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
        RuntimeError: If attempting to run this model with a
            backend that does not support separable convolutions.
    """
    return NASNet(
        input_shape,
        penultimate_filters=4032,
        num_blocks=6,
        stem_block_filters=96,
        skip_reduction=True,
        filter_multiplier=2,
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        pooling=pooling,
        classes=classes,
        default_size=331,
        classifier_activation=classifier_activation,
    )
def _separable_conv_block(
    ip, filters, kernel_size=(3, 3), strides=(1, 1), block_id=None
):
    """Adds 2 blocks of [relu-separable conv-batchnorm].
    Args:
        ip: Input tensor
        filters: Number of output filters per layer
        kernel_size: Kernel size of separable convolutions
        strides: Strided convolution for downsampling
        block_id: String block_id
    Returns:
        A Keras tensor
    """
    channel_dim = 1 if backend.image_data_format() == "channels_first" else -1
    with backend.name_scope(f"separable_conv_block_{block_id}"):
        x = layers.Activation("relu")(ip)
        if strides == (2, 2):
            x = layers.ZeroPadding2D(
                padding=imagenet_utils.correct_pad(x, kernel_size),
                name=f"separable_conv_1_pad_{block_id}",
            )(x)
            conv_pad = "valid"
        else:
            conv_pad = "same"
        x = layers.SeparableConv2D(
            filters,
            kernel_size,
            strides=strides,
            name=f"separable_conv_1_{block_id}",
            padding=conv_pad,
            use_bias=False,
            kernel_initializer="he_normal",
        )(x)
        x = layers.BatchNormalization(
            axis=channel_dim,
            momentum=0.9997,
            epsilon=1e-3,
            name=f"separable_conv_1_bn_{block_id}",
        )(x)
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(
            filters,
            kernel_size,
            name=f"separable_conv_2_{block_id}",
            padding="same",
            use_bias=False,
            kernel_initializer="he_normal",
        )(x)
        x = layers.BatchNormalization(
            axis=channel_dim,
            momentum=0.9997,
            epsilon=1e-3,
            name=f"separable_conv_2_bn_{block_id}",
        )(x)
    return x
def _adjust_block(p, ip, filters, block_id=None):
    """Adjusts the input `previous path` to match the shape of the `input`.
    Used in situations where the output number of filters needs to be changed.
    Args:
        p: Input tensor which needs to be modified
        ip: Input tensor whose shape needs to be matched
        filters: Number of output filters to be matched
        block_id: String block_id
    Returns:
        Adjusted Keras tensor
    """
    channel_dim = 1 if backend.image_data_format() == "channels_first" else -1
    img_dim = 2 if backend.image_data_format() == "channels_first" else -2
    ip_shape = backend.int_shape(ip)
    if p is not None:
        p_shape = backend.int_shape(p)
    with backend.name_scope("adjust_block"):
        if p is None:
            p = ip
        elif p_shape[img_dim] != ip_shape[img_dim]:
            with backend.name_scope(f"adjust_reduction_block_{block_id}"):
                p = layers.Activation("relu", name=f"adjust_relu_1_{block_id}")(
                    p
                )
                p1 = layers.AveragePooling2D(
                    (1, 1),
                    strides=(2, 2),
                    padding="valid",
                    name=f"adjust_avg_pool_1_{block_id}",
                )(p)
                p1 = layers.Conv2D(
                    filters // 2,
                    (1, 1),
                    padding="same",
                    use_bias=False,
                    name=f"adjust_conv_1_{block_id}",
                    kernel_initializer="he_normal",
                )(p1)
                p2 = layers.ZeroPadding2D(padding=((0, 1), (0, 1)))(p)
                p2 = layers.Cropping2D(cropping=((1, 0), (1, 0)))(p2)
                p2 = layers.AveragePooling2D(
                    (1, 1),
                    strides=(2, 2),
                    padding="valid",
                    name=f"adjust_avg_pool_2_{block_id}",
                )(p2)
                p2 = layers.Conv2D(
                    filters // 2,
                    (1, 1),
                    padding="same",
                    use_bias=False,
                    name=f"adjust_conv_2_{block_id}",
                    kernel_initializer="he_normal",
                )(p2)
                p = layers.concatenate([p1, p2], axis=channel_dim)
                p = layers.BatchNormalization(
                    axis=channel_dim,
                    momentum=0.9997,
                    epsilon=1e-3,
                    name=f"adjust_bn_{block_id}",
                )(p)
        elif p_shape[channel_dim] != filters:
            with backend.name_scope(f"adjust_projection_block_{block_id}"):
                p = layers.Activation("relu")(p)
                p = layers.Conv2D(
                    filters,
                    (1, 1),
                    strides=(1, 1),
                    padding="same",
                    name=f"adjust_conv_projection_{block_id}",
                    use_bias=False,
                    kernel_initializer="he_normal",
                )(p)
                p = layers.BatchNormalization(
                    axis=channel_dim,
                    momentum=0.9997,
                    epsilon=1e-3,
                    name=f"adjust_bn_{block_id}",
                )(p)
    return p
def _normal_a_cell(ip, p, filters, block_id=None):
    """Adds a Normal cell for NASNet-A (Fig. 4 in the paper).
    Args:
        ip: Input tensor `x`
        p: Input tensor `p`
        filters: Number of output filters
        block_id: String block_id
    Returns:
        A Keras tensor
    """
    channel_dim = 1 if backend.image_data_format() == "channels_first" else -1
    with backend.name_scope(f"normal_A_block_{block_id}"):
        p = _adjust_block(p, ip, filters, block_id)
        h = layers.Activation("relu")(ip)
        h = layers.Conv2D(
            filters,
            (1, 1),
            strides=(1, 1),
            padding="same",
            name=f"normal_conv_1_{block_id}",
            use_bias=False,
            kernel_initializer="he_normal",
        )(h)
        h = layers.BatchNormalization(
            axis=channel_dim,
            momentum=0.9997,
            epsilon=1e-3,
            name=f"normal_bn_1_{block_id}",
        )(h)
        with backend.name_scope("block_1"):
            x1_1 = _separable_conv_block(
                h,
                filters,
                kernel_size=(5, 5),
                block_id=f"normal_left1_{block_id}",
            )
            x1_2 = _separable_conv_block(
                p, filters, block_id=f"normal_right1_{block_id}"
            )
            x1 = layers.add([x1_1, x1_2], name=f"normal_add_1_{block_id}")
        with backend.name_scope("block_2"):
            x2_1 = _separable_conv_block(
                p, filters, (5, 5), block_id=f"normal_left2_{block_id}"
            )
            x2_2 = _separable_conv_block(
                p, filters, (3, 3), block_id=f"normal_right2_{block_id}"
            )
            x2 = layers.add([x2_1, x2_2], name=f"normal_add_2_{block_id}")
        with backend.name_scope("block_3"):
            x3 = layers.AveragePooling2D(
                (3, 3),
                strides=(1, 1),
                padding="same",
                name=f"normal_left3_{block_id}",
            )(h)
            x3 = layers.add([x3, p], name=f"normal_add_3_{block_id}")
        with backend.name_scope("block_4"):
            x4_1 = layers.AveragePooling2D(
                (3, 3),
                strides=(1, 1),
                padding="same",
                name=f"normal_left4_{block_id}",
            )(p)
            x4_2 = layers.AveragePooling2D(
                (3, 3),
                strides=(1, 1),
                padding="same",
                name=f"normal_right4_{block_id}",
            )(p)
            x4 = layers.add([x4_1, x4_2], name=f"normal_add_4_{block_id}")
        with backend.name_scope("block_5"):
            x5 = _separable_conv_block(
                h, filters, block_id=f"normal_left5_{block_id}"
            )
            x5 = layers.add([x5, h], name=f"normal_add_5_{block_id}")
        x = layers.concatenate(
            [p, x1, x2, x3, x4, x5],
            axis=channel_dim,
            name=f"normal_concat_{block_id}",
        )
    return x, ip
def _reduction_a_cell(ip, p, filters, block_id=None):
    """Adds a Reduction cell for NASNet-A (Fig. 4 in the paper).
    Args:
      ip: Input tensor `x`
      p: Input tensor `p`
      filters: Number of output filters
      block_id: String block_id
    Returns:
      A Keras tensor
    """
    channel_dim = 1 if backend.image_data_format() == "channels_first" else -1
    with backend.name_scope(f"reduction_A_block_{block_id}"):
        p = _adjust_block(p, ip, filters, block_id)
        h = layers.Activation("relu")(ip)
        h = layers.Conv2D(
            filters,
            (1, 1),
            strides=(1, 1),
            padding="same",
            name=f"reduction_conv_1_{block_id}",
            use_bias=False,
            kernel_initializer="he_normal",
        )(h)
        h = layers.BatchNormalization(
            axis=channel_dim,
            momentum=0.9997,
            epsilon=1e-3,
            name=f"reduction_bn_1_{block_id}",
        )(h)
        h3 = layers.ZeroPadding2D(
            padding=imagenet_utils.correct_pad(h, 3),
            name=f"reduction_pad_1_{block_id}",
        )(h)
        with backend.name_scope("block_1"):
            x1_1 = _separable_conv_block(
                h,
                filters,
                (5, 5),
                strides=(2, 2),
                block_id=f"reduction_left1_{block_id}",
            )
            x1_2 = _separable_conv_block(
                p,
                filters,
                (7, 7),
                strides=(2, 2),
                block_id=f"reduction_right1_{block_id}",
            )
            x1 = layers.add([x1_1, x1_2], name=f"reduction_add_1_{block_id}")
        with backend.name_scope("block_2"):
            x2_1 = layers.MaxPooling2D(
                (3, 3),
                strides=(2, 2),
                padding="valid",
                name=f"reduction_left2_{block_id}",
            )(h3)
            x2_2 = _separable_conv_block(
                p,
                filters,
                (7, 7),
                strides=(2, 2),
                block_id=f"reduction_right2_{block_id}",
            )
            x2 = layers.add([x2_1, x2_2], name=f"reduction_add_2_{block_id}")
        with backend.name_scope("block_3"):
            x3_1 = layers.AveragePooling2D(
                (3, 3),
                strides=(2, 2),
                padding="valid",
                name=f"reduction_left3_{block_id}",
            )(h3)
            x3_2 = _separable_conv_block(
                p,
                filters,
                (5, 5),
                strides=(2, 2),
                block_id=f"reduction_right3_{block_id}",
            )
            x3 = layers.add([x3_1, x3_2], name=f"reduction_add3_{block_id}")
        with backend.name_scope("block_4"):
            x4 = layers.AveragePooling2D(
                (3, 3),
                strides=(1, 1),
                padding="same",
                name=f"reduction_left4_{block_id}",
            )(x1)
            x4 = layers.add([x2, x4])
        with backend.name_scope("block_5"):
            x5_1 = _separable_conv_block(
                x1, filters, (3, 3), block_id=f"reduction_left4_{block_id}"
            )
            x5_2 = layers.MaxPooling2D(
                (3, 3),
                strides=(2, 2),
                padding="valid",
                name=f"reduction_right5_{block_id}",
            )(h3)
            x5 = layers.add([x5_1, x5_2], name=f"reduction_add4_{block_id}")
        x = layers.concatenate(
            [x2, x3, x4, x5],
            axis=channel_dim,
            name=f"reduction_concat_{block_id}",
        )
        return x, ip
