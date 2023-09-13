@keras_export("keras.backend.conv1d")
@tf.__internal__.dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def conv1d(
    x, kernel, strides=1, padding="valid", data_format=None, dilation_rate=1
