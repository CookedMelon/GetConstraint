@tf_export("strings.to_hash_bucket", v1=[])
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def string_to_hash_bucket(input, num_buckets, name=None):
  # pylint: disable=line-too-long
  r"""Converts each string in the input Tensor to its hash mod by a number of buckets.
  The hash function is deterministic on the content of the string within the
  process.
  Note that the hash function may change from time to time.
  This functionality will be deprecated and it's recommended to use
  `tf.strings.to_hash_bucket_fast()` or `tf.strings.to_hash_bucket_strong()`.
  Examples:
  >>> tf.strings.to_hash_bucket(["Hello", "TensorFlow", "2.x"], 3)
  <tf.Tensor: shape=(3,), dtype=int64, numpy=array([2, 0, 1])>
  Args:
    input: A `Tensor` of type `string`.
    num_buckets: An `int` that is `>= 1`. The number of buckets.
    name: A name for the operation (optional).
  Returns:
    A `Tensor` of type `int64`.
  """
  # pylint: enable=line-too-long
  return gen_string_ops.string_to_hash_bucket(input, num_buckets, name)
