@tf_export("test_op")
@dispatch.add_dispatch_support
def test_op(x, y, z):
  """A fake op for testing dispatch of Python ops."""
  return x + (2 * y) + (3 * z)
