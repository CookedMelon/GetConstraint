@keras_export(
    "keras.applications.regnet.RegNetX160", "keras.applications.RegNetX160"
)
def RegNetX160(
    model_name="regnetx160",
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
        MODEL_CONFIGS["x160"]["depths"],
        MODEL_CONFIGS["x160"]["widths"],
        MODEL_CONFIGS["x160"]["group_width"],
        MODEL_CONFIGS["x160"]["block_type"],
        MODEL_CONFIGS["x160"]["default_size"],
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
