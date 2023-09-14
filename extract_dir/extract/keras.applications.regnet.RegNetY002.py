@keras_export(
    "keras.applications.regnet.RegNetY002", "keras.applications.RegNetY002"
)
def RegNetY002(
    model_name="regnety002",
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
        MODEL_CONFIGS["y002"]["depths"],
        MODEL_CONFIGS["y002"]["widths"],
        MODEL_CONFIGS["y002"]["group_width"],
        MODEL_CONFIGS["y002"]["block_type"],
        MODEL_CONFIGS["y002"]["default_size"],
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
