@tf_export("test.with_eager_op_as_function")
def with_eager_op_as_function(cls=None, only_as_function=False):  # pylint: disable=unused-argument
  """Returns the same class. This will be removed once all usages are removed.
  Args:
    cls: class to decorate.
    only_as_function: unused argument.
  Returns:
    cls
  """
  def decorator(cls):
    return cls
  if cls is not None:
    return decorator(cls)
  return decorator
