@keras_export("keras.backend.pool3d")
@tf.__internal__.dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def pool3d(
    x,
    pool_size,
    strides=(1, 1, 1),
    padding="valid",
    data_format=None,
    pool_mode="max",
