@keras_export("keras.backend.sparse_categorical_crossentropy")
@tf.__internal__.dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def sparse_categorical_crossentropy(
    target, output, from_logits=False, axis=-1, ignore_class=None
