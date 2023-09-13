@keras_export("keras.backend.categorical_focal_crossentropy")
@tf.__internal__.dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def categorical_focal_crossentropy(
    target,
    output,
    alpha=0.25,
    gamma=2.0,
    from_logits=False,
    axis=-1,
