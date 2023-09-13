@keras_export("keras.backend.name_scope", v1=[])
@doc_controls.do_not_generate_docs
def name_scope(name):
    """A context manager for use when defining a Python op.
    This context manager pushes a name scope, which will make the name of all
    operations added within it have a prefix.
    For example, to define a new Python op called `my_op`:
    def my_op(a):
      with tf.name_scope("MyOp") as scope:
        a = tf.convert_to_tensor(a, name="a")
        # Define some computation that uses `a`.
        return foo_op(..., name=scope)
    When executed, the Tensor `a` will have the name `MyOp/a`.
    Args:
      name: The prefix to use on all names created within the name scope.
    Returns:
      Name scope context manager.
    """
    return tf.name_scope(name)
