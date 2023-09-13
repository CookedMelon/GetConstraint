@tf_export("distribute.experimental.CommunicationOptions")
class _OptionsExported(object):
  """Options for cross device communications like All-reduce.
  This can be passed to methods like
  `tf.distribute.get_replica_context().all_reduce()` to optimize collective
  operation performance. Note that these are only hints, which may or may not
  change the actual behavior. Some options only apply to certain strategy and
  are ignored by others.
  One common optimization is to break gradients all-reduce into multiple packs
  so that weight updates can overlap with gradient all-reduce.
  Examples:
  ```python
  options = tf.distribute.experimental.CommunicationOptions(
      bytes_per_pack=50 * 1024 * 1024,
      timeout_seconds=120.0,
      implementation=tf.distribute.experimental.CommunicationImplementation.NCCL
  )
  grads = tf.distribute.get_replica_context().all_reduce(
      'sum', grads, options=options)
  optimizer.apply_gradients(zip(grads, vars),
      experimental_aggregate_gradients=False)
  ```
  """
  def __new__(cls, *args, **kwargs):
    # We expose a dummy class so that we can separate internal and public APIs.
    # Note that __init__ won't be called on the returned object if it's a
    # different class [1].
    # [1] https://docs.python.org/3/reference/datamodel.html#object.__new__
    return Options(*args, **kwargs)
  def __init__(self,
               bytes_per_pack=0,
               timeout_seconds=None,
               implementation=CommunicationImplementation.AUTO):
    """Creates a CollectiveHints.
    Args:
      bytes_per_pack: a non-negative integer. Breaks collective operations into
        packs of certain size. If it's zero, the value is determined
        automatically. This hint is respected by all multi-replica strategies
        except `TPUStrategy`.
      timeout_seconds: a float or None, timeout in seconds. If not None, the
        collective raises `tf.errors.DeadlineExceededError` if it takes longer
        than this timeout. Zero disables timeout. This can be useful when
        debugging hanging issues.  This should only be used for debugging since
        it creates a new thread for each collective, i.e. an overhead of
        `timeout_seconds * num_collectives_per_second` more threads. This only
        works for `tf.distribute.experimental.MultiWorkerMirroredStrategy`.
      implementation: a
        `tf.distribute.experimental.CommunicationImplementation`. This is a hint
        on the preferred communication implementation. Possible values include
        `AUTO`, `RING`, and `NCCL`. NCCL is generally more performant for GPU,
        but doesn't work for CPU. This only works for
        `tf.distribute.experimental.MultiWorkerMirroredStrategy`.
    Raises:
      ValueError: When arguments have invalid value.
    """
    pass
