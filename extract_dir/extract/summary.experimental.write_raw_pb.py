@tf_export("summary.experimental.write_raw_pb", v1=[])
def write_raw_pb(tensor, step=None, name=None):
  """Writes a summary using raw `tf.compat.v1.Summary` protocol buffers.
  Experimental: this exists to support the usage of V1-style manual summary
  writing (via the construction of a `tf.compat.v1.Summary` protocol buffer)
  with the V2 summary writing API.
  Args:
    tensor: the string Tensor holding one or more serialized `Summary` protobufs
    step: Explicit `int64`-castable monotonic step value for this summary. If
      omitted, this defaults to `tf.summary.experimental.get_step()`, which must
      not be None.
    name: Optional string name for this op.
  Returns:
    True on success, or false if no summary was written because no default
    summary writer was available.
  Raises:
    ValueError: if a default writer exists, but no step was provided and
      `tf.summary.experimental.get_step()` is None.
  """
  with ops.name_scope(name, "write_raw_pb") as scope:
    if _summary_state.writer is None:
      return constant_op.constant(False)
    if step is None:
      step = get_step()
      if step is None:
        raise ValueError("No step set. Please specify one either through the "
                         "`step` argument or through "
                         "tf.summary.experimental.set_step()")
    def record():
      """Record the actual summary and return True."""
      # Note the identity to move the tensor to the CPU.
      with ops.device("cpu:0"):
        raw_summary_op = gen_summary_ops.write_raw_proto_summary(
            _summary_state.writer._resource,  # pylint: disable=protected-access
            step,
            array_ops.identity(tensor),
            name=scope)
        with ops.control_dependencies([raw_summary_op]):
          return constant_op.constant(True)
    with ops.device("cpu:0"):
      op = smart_cond.smart_cond(
          should_record_summaries(), record, _nothing, name="summary_cond")
      if not context.executing_eagerly():
        ops.add_to_collection(ops.GraphKeys._SUMMARY_COLLECTION, op)  # pylint: disable=protected-access
      return op
