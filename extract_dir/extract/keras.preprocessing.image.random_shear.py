@keras_export("keras.preprocessing.image.random_shear")
def random_shear(
    x,
    intensity,
    row_axis=1,
    col_axis=2,
    channel_axis=0,
    fill_mode="nearest",
    cval=0.0,
    interpolation_order=1,
