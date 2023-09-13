@keras_export("keras.preprocessing.image.random_channel_shift")
def random_channel_shift(x, intensity_range, channel_axis=0):
    """Performs a random channel shift.
    Args:
        x: Input tensor. Must be 3D.
        intensity_range: Transformation intensity.
        channel_axis: Index of axis for channels in the input tensor.
    Returns:
        Numpy image tensor.
    """
    intensity = np.random.uniform(-intensity_range, intensity_range)
    return apply_channel_shift(x, intensity, channel_axis=channel_axis)
