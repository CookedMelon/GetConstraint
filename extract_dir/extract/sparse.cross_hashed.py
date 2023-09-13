@tf_export("sparse.cross_hashed")
def sparse_cross_hashed(inputs, num_buckets=0, hash_key=None, name=None):
  """Generates hashed sparse cross from a list of sparse and dense tensors.
  For example, if the inputs are
      * inputs[0]: SparseTensor with shape = [2, 2]
        [0, 0]: "a"
        [1, 0]: "b"
        [1, 1]: "c"
      * inputs[1]: SparseTensor with shape = [2, 1]
        [0, 0]: "d"
        [1, 0]: "e"
      * inputs[2]: Tensor [["f"], ["g"]]
  then the output will be:
      shape = [2, 2]
      [0, 0]: FingerprintCat64(
                  Fingerprint64("f"), FingerprintCat64(
                      Fingerprint64("d"), Fingerprint64("a")))
      [1, 0]: FingerprintCat64(
                  Fingerprint64("g"), FingerprintCat64(
                      Fingerprint64("e"), Fingerprint64("b")))
      [1, 1]: FingerprintCat64(
                  Fingerprint64("g"), FingerprintCat64(
                      Fingerprint64("e"), Fingerprint64("c")))
  Args:
    inputs: An iterable of `Tensor` or `SparseTensor`.
    num_buckets: An `int` that is `>= 0`.
      output = hashed_value%num_buckets if num_buckets > 0 else hashed_value.
    hash_key: Integer hash_key that will be used by the `FingerprintCat64`
      function. If not given, will use a default key.
    name: Optional name for the op.
  Returns:
    A `SparseTensor` of type `int64`.
  """
  return _sparse_cross_internal(
      inputs=inputs,
      hashed_output=True,
      num_buckets=num_buckets,
      hash_key=hash_key,
      name=name)
