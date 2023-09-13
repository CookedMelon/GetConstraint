@keras_export("keras.backend.binary_focal_crossentropy")
@tf.__internal__.dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def binary_focal_crossentropy(
    target,
    output,
    apply_class_balancing=False,
    alpha=0.25,
    gamma=2.0,
    from_logits=False,
