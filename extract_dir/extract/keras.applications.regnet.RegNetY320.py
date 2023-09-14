@keras_export(
    "keras.applications.regnet.RegNetY320", "keras.applications.RegNetY320"
)
def RegNetY320(
    model_name="regnety320",
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
        MODEL_CONFIGS["y320"]["depths"],
        MODEL_CONFIGS["y320"]["widths"],
        MODEL_CONFIGS["y320"]["group_width"],
        MODEL_CONFIGS["y320"]["block_type"],
        MODEL_CONFIGS["y320"]["default_size"],
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
RegNetX002.__doc__ = BASE_DOCSTRING.format(name="RegNetX002")
RegNetX004.__doc__ = BASE_DOCSTRING.format(name="RegNetX004")
RegNetX006.__doc__ = BASE_DOCSTRING.format(name="RegNetX006")
RegNetX008.__doc__ = BASE_DOCSTRING.format(name="RegNetX008")
RegNetX016.__doc__ = BASE_DOCSTRING.format(name="RegNetX016")
RegNetX032.__doc__ = BASE_DOCSTRING.format(name="RegNetX032")
RegNetX040.__doc__ = BASE_DOCSTRING.format(name="RegNetX040")
RegNetX064.__doc__ = BASE_DOCSTRING.format(name="RegNetX064")
RegNetX080.__doc__ = BASE_DOCSTRING.format(name="RegNetX080")
RegNetX120.__doc__ = BASE_DOCSTRING.format(name="RegNetX120")
RegNetX160.__doc__ = BASE_DOCSTRING.format(name="RegNetX160")
RegNetX320.__doc__ = BASE_DOCSTRING.format(name="RegNetX320")
RegNetY002.__doc__ = BASE_DOCSTRING.format(name="RegNetY002")
RegNetY004.__doc__ = BASE_DOCSTRING.format(name="RegNetY004")
RegNetY006.__doc__ = BASE_DOCSTRING.format(name="RegNetY006")
RegNetY008.__doc__ = BASE_DOCSTRING.format(name="RegNetY008")
RegNetY016.__doc__ = BASE_DOCSTRING.format(name="RegNetY016")
RegNetY032.__doc__ = BASE_DOCSTRING.format(name="RegNetY032")
RegNetY040.__doc__ = BASE_DOCSTRING.format(name="RegNetY040")
RegNetY064.__doc__ = BASE_DOCSTRING.format(name="RegNetY064")
RegNetY080.__doc__ = BASE_DOCSTRING.format(name="RegNetY080")
RegNetY120.__doc__ = BASE_DOCSTRING.format(name="RegNetY120")
RegNetY160.__doc__ = BASE_DOCSTRING.format(name="RegNetY160")
RegNetY320.__doc__ = BASE_DOCSTRING.format(name="RegNetY320")
