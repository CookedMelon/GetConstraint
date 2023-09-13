@keras_export("keras.backend.resize_images")
@tf.__internal__.dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def resize_images(
    x, height_factor, width_factor, data_format, interpolation="nearest"
