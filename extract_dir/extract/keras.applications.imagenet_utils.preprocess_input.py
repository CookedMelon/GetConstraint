@keras_export("keras.applications.imagenet_utils.preprocess_input")
def preprocess_input(x, data_format=None, mode="caffe"):
    """Preprocesses a tensor or Numpy array encoding a batch of images."""
    if mode not in {"caffe", "tf", "torch"}:
        raise ValueError(
            "Expected mode to be one of `caffe`, `tf` or `torch`. "
            f"Received: mode={mode}"
        )
    if data_format is None:
        data_format = backend.image_data_format()
    elif data_format not in {"channels_first", "channels_last"}:
        raise ValueError(
            "Expected data_format to be one of `channels_first` or "
            f"`channels_last`. Received: data_format={data_format}"
        )
    if isinstance(x, np.ndarray):
        return _preprocess_numpy_input(x, data_format=data_format, mode=mode)
    else:
        return _preprocess_symbolic_input(x, data_format=data_format, mode=mode)
