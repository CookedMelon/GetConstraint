@keras_export("keras.backend.depthwise_conv2d")
@tf.__internal__.dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def depthwise_conv2d(
    x,
    depthwise_kernel,
    strides=(1, 1),
    padding="valid",
    data_format=None,
    dilation_rate=(1, 1),
