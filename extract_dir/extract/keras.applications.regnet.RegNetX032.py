@keras_export(
    "keras.applications.regnet.RegNetX032", "keras.applications.RegNetX032"
)
def RegNetX032(
    model_name="regnetx032",
    include_top=True,
    include_preprocessing=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
):
    return RegNet(
        MODEL_CONFIGS["x032"]["depths"],
        MODEL_CONFIGS["x032"]["widths"],
        MODEL_CONFIGS["x032"]["group_width"],
        MODEL_CONFIGS["x032"]["block_type"],
        MODEL_CONFIGS["x032"]["default_size"],
        model_name=model_name,
        include_top=include_top,
        include_preprocessing=include_preprocessing,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
    )
