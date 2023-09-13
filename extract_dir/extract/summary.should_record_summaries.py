@tf_export("summary.should_record_summaries", v1=[])
def should_record_summaries():
  """Returns boolean Tensor which is True if summaries will be recorded.
  If no default summary writer is currently registered, this always returns
  False. Otherwise, this reflects the recording condition has been set via
  `tf.summary.record_if()` (except that it may return False for some replicas
  when using `tf.distribute.Strategy`). If no recording condition is active,
  it defaults to True.
  """
  return _should_record_summaries_internal(default_state=True)
