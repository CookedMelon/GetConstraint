@keras_export("keras.backend.batch_get_value")
@tf.__internal__.dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def batch_get_value(tensors):
    """Returns the value of more than one tensor variable.
    Args:
        tensors: list of ops to run.
    Returns:
        A list of Numpy arrays.
    Raises:
        RuntimeError: If this method is called inside defun.
    """
    if tf.executing_eagerly():
        return [x.numpy() for x in tensors]
    elif tf.inside_function():
        raise RuntimeError("Cannot get value inside Tensorflow graph function.")
    if tensors:
        return get_session(tensors).run(tensors)
    else:
        return []
