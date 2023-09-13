@keras_export("keras.backend.pool2d")
@tf.__internal__.dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def pool2d(
    x,
    pool_size,
    strides=(1, 1),
    padding="valid",
    data_format=None,
    pool_mode="max",
