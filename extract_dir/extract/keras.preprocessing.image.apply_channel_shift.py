@keras_export("keras.preprocessing.image.apply_channel_shift")
def apply_channel_shift(x, intensity, channel_axis=0):
    """Performs a channel shift.
    Args:
        x: Input tensor. Must be 3D.
        intensity: Transformation intensity.
        channel_axis: Index of axis for channels in the input tensor.
    Returns:
        Numpy image tensor.
    """
    x = np.rollaxis(x, channel_axis, 0)
    min_x, max_x = np.min(x), np.max(x)
    channel_images = [
        np.clip(x_channel + intensity, min_x, max_x) for x_channel in x
    ]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x
