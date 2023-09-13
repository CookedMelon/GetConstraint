@keras_export("keras.backend.local_conv2d")
@tf.__internal__.dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def local_conv2d(
    inputs, kernel, kernel_size, strides, output_shape, data_format=None
