@keras_export("keras.backend.function")
@doc_controls.do_not_generate_docs
def function(inputs, outputs, updates=None, name=None, **kwargs):
    """Instantiates a Keras function.
    Args:
        inputs: List of placeholder tensors.
        outputs: List of output tensors.
        updates: List of update ops.
        name: String, name of function.
        **kwargs: Passed to `tf.Session.run`.
    Returns:
        Output values as Numpy arrays.
    Raises:
        ValueError: if invalid kwargs are passed in or if in eager execution.
    """
    if tf.compat.v1.executing_eagerly_outside_functions():
        if kwargs:
            raise ValueError(
                "Session keyword arguments are not supported during "
                "eager execution. You passed: %s" % (kwargs,)
            )
        if updates:
            raise ValueError(
                "`updates` argument is not supported during "
                "eager execution. You passed: %s" % (updates,)
            )
        from keras import models
        model = models.Model(inputs=inputs, outputs=outputs)
        wrap_outputs = isinstance(outputs, list) and len(outputs) == 1
        def func(model_inputs):
            outs = model(model_inputs)
            if wrap_outputs:
                outs = [outs]
            return tf_utils.sync_to_numpy_or_python_type(outs)
        return func
    if kwargs:
        for key in kwargs:
            if key not in tf_inspect.getfullargspec(tf.compat.v1.Session.run)[
                0
            ] and key not in ["inputs", "outputs", "updates", "name"]:
                msg = (
                    'Invalid argument "%s" passed to K.function with '
                    "TensorFlow backend" % key
                )
                raise ValueError(msg)
    return GraphExecutionFunction(
        inputs, outputs, updates=updates, name=name, **kwargs
    )
