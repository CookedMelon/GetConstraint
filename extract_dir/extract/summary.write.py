@tf_export("summary.write", v1=[])
def write(tag, tensor, step=None, metadata=None, name=None):
  """Writes a generic summary to the default SummaryWriter if one exists.
  This exists primarily to support the definition of type-specific summary ops
  like scalar() and image(), and is not intended for direct use unless defining
  a new type-specific summary op.
  Args:
    tag: string tag used to identify the summary (e.g. in TensorBoard), usually
      generated with `tf.summary.summary_scope`
    tensor: the Tensor holding the summary data to write or a callable that
      returns this Tensor. If a callable is passed, it will only be called when
      a default SummaryWriter exists and the recording condition specified by
      `record_if()` is met.
    step: Explicit `int64`-castable monotonic step value for this summary. If
      omitted, this defaults to `tf.summary.experimental.get_step()`, which must
      not be None.
    metadata: Optional SummaryMetadata, as a proto or serialized bytes
    name: Optional string name for this op.
  Returns:
    True on success, or false if no summary was written because no default
    summary writer was available.
  Raises:
    ValueError: if a default writer exists, but no step was provided and
      `tf.summary.experimental.get_step()` is None.
  """
  with ops.name_scope(name, "write_summary") as scope:
    if _summary_state.writer is None:
      return constant_op.constant(False)
    if step is None:
      step = get_step()
    if metadata is None:
      serialized_metadata = b""
    elif hasattr(metadata, "SerializeToString"):
      serialized_metadata = metadata.SerializeToString()
    else:
      serialized_metadata = metadata
    def record():
      """Record the actual summary and return True."""
      if step is None:
        raise ValueError("No step set. Please specify one either through the "
                         "`step` argument or through "
                         "tf.summary.experimental.set_step()")
      # Note the identity to move the tensor to the CPU.
      with ops.device("cpu:0"):
        summary_tensor = tensor() if callable(tensor) else array_ops.identity(
            tensor)
        write_summary_op = gen_summary_ops.write_summary(
            _summary_state.writer._resource,  # pylint: disable=protected-access
            step,
            summary_tensor,
            tag,
            serialized_metadata,
            name=scope)
        with ops.control_dependencies([write_summary_op]):
          return constant_op.constant(True)
    op = smart_cond.smart_cond(
        should_record_summaries(), record, _nothing, name="summary_cond")
    if not context.executing_eagerly():
      ops.add_to_collection(ops.GraphKeys._SUMMARY_COLLECTION, op)  # pylint: disable=protected-access
    return op
