@keras_export("keras.backend.update")
@doc_controls.do_not_generate_docs
def update(x, new_x):
    return tf.compat.v1.assign(x, new_x)
