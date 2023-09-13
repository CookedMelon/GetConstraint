@keras_export("keras.applications.convnext.preprocess_input")
def preprocess_input(x, data_format=None):
    """A placeholder method for backward compatibility.
    The preprocessing logic has been included in the convnext model
    implementation. Users are no longer required to call this method to
    normalize the input data. This method does nothing and only kept as a
    placeholder to align the API surface between old and new version of model.
    Args:
      x: A floating point `numpy.array` or a `tf.Tensor`.
      data_format: Optional data format of the image tensor/array. Defaults to
        None, in which case the global setting
        `tf.keras.backend.image_data_format()` is used (unless you changed it,
        it defaults to "channels_last").{mode}
    Returns:
      Unchanged `numpy.array` or `tf.Tensor`.
    """
    return x
