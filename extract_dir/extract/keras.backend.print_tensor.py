@keras_export("keras.backend.print_tensor")
@tf.__internal__.dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def print_tensor(x, message="", summarize=3):
    """Prints `message` and the tensor value when evaluated.
    Note that `print_tensor` returns a new tensor identical to `x`
    which should be used in the following code. Otherwise the
    print operation is not taken into account during evaluation.
    Example:
    >>> x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    >>> tf.keras.backend.print_tensor(x)
    <tf.Tensor: shape=(2, 2), dtype=float32, numpy=
      array([[1., 2.],
             [3., 4.]], dtype=float32)>
    Args:
        x: Tensor to print.
        message: Message to print jointly with the tensor.
        summarize: The first and last `summarize` elements within each dimension
            are recursively printed per Tensor. If None, then the first 3 and
            last 3 elements of each dimension are printed for each tensor. If
            set to -1, it will print all elements of every tensor.
    Returns:
        The same tensor `x`, unchanged.
    """
    if isinstance(x, tf.Tensor) and hasattr(x, "graph"):
        with get_graph().as_default():
            op = tf.print(
                message, x, output_stream=sys.stdout, summarize=summarize
            )
            with tf.control_dependencies([op]):
                return tf.identity(x)
    else:
        tf.print(message, x, output_stream=sys.stdout, summarize=summarize)
        return x
