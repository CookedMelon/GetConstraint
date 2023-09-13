@keras_export("keras.backend.conv2d_transpose")
@tf.__internal__.dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def conv2d_transpose(
    x,
    kernel,
    output_shape,
    strides=(1, 1),
    padding="valid",
    data_format=None,
    dilation_rate=(1, 1),
