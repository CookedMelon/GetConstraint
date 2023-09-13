@tf_export("data.experimental.ThreadingOptions", "data.ThreadingOptions")
class ThreadingOptions(options_lib.OptionsBase):
  """Represents options for dataset threading.
  You can set the threading options of a dataset through the
  `threading` property of `tf.data.Options`; the property is
  an instance of `tf.data.ThreadingOptions`.
  ```python
  options = tf.data.Options()
  options.threading.private_threadpool_size = 10
  dataset = dataset.with_options(options)
  ```
  """
  max_intra_op_parallelism = options_lib.create_option(
      name="max_intra_op_parallelism",
      ty=int,
      docstring=
      "If set, it overrides the maximum degree of intra-op parallelism.")
  private_threadpool_size = options_lib.create_option(
      name="private_threadpool_size",
      ty=int,
      docstring=
      "If set, the dataset will use a private threadpool of the given size. "
      "The value 0 can be used to indicate that the threadpool size should be "
      "determined at runtime based on the number of available CPU cores.")
  def _to_proto(self):
    pb = dataset_options_pb2.ThreadingOptions()
    if self.max_intra_op_parallelism is not None:
      pb.max_intra_op_parallelism = self.max_intra_op_parallelism
    if self.private_threadpool_size is not None:
      pb.private_threadpool_size = self.private_threadpool_size
    return pb
  def _from_proto(self, pb):
    if pb.WhichOneof("optional_max_intra_op_parallelism") is not None:
      self.max_intra_op_parallelism = pb.max_intra_op_parallelism
    if pb.WhichOneof("optional_private_threadpool_size") is not None:
      self.private_threadpool_size = pb.private_threadpool_size
