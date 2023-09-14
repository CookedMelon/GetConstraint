@tf_export(
    "queue.RandomShuffleQueue",
    v1=["queue.RandomShuffleQueue",
        "io.RandomShuffleQueue", "RandomShuffleQueue"])
@deprecation.deprecated_endpoints(
    ["io.RandomShuffleQueue", "RandomShuffleQueue"])
class RandomShuffleQueue(QueueBase):
  """A queue implementation that dequeues elements in a random order.
  See `tf.queue.QueueBase` for a description of the methods on
  this class.
  """
  def __init__(self,
               capacity,
               min_after_dequeue,
               dtypes,
               shapes=None,
               names=None,
               seed=None,
               shared_name=None,
               name="random_shuffle_queue"):
    """Create a queue that dequeues elements in a random order.
    A `RandomShuffleQueue` has bounded capacity; supports multiple
    concurrent producers and consumers; and provides exactly-once
    delivery.
    A `RandomShuffleQueue` holds a list of up to `capacity`
    elements. Each element is a fixed-length tuple of tensors whose
    dtypes are described by `dtypes`, and whose shapes are optionally
    described by the `shapes` argument.
    If the `shapes` argument is specified, each component of a queue
    element must have the respective fixed shape. If it is
    unspecified, different queue elements may have different shapes,
    but the use of `dequeue_many` is disallowed.
    The `min_after_dequeue` argument allows the caller to specify a
    minimum number of elements that will remain in the queue after a
    `dequeue` or `dequeue_many` operation completes, to ensure a
    minimum level of mixing of elements. This invariant is maintained
    by blocking those operations until sufficient elements have been
    enqueued. The `min_after_dequeue` argument is ignored after the
    queue has been closed.
    Args:
      capacity: An integer. The upper bound on the number of elements
        that may be stored in this queue.
      min_after_dequeue: An integer (described above).
      dtypes:  A list of `DType` objects. The length of `dtypes` must equal
        the number of tensors in each queue element.
      shapes: (Optional.) A list of fully-defined `TensorShape` objects
        with the same length as `dtypes`, or `None`.
      names: (Optional.) A list of string naming the components in the queue
        with the same length as `dtypes`, or `None`.  If specified the dequeue
        methods return a dictionary with the names as keys.
      seed: A Python integer. Used to create a random seed. See
        `tf.compat.v1.set_random_seed`
        for behavior.
      shared_name: (Optional.) If non-empty, this queue will be shared under
        the given name across multiple sessions.
      name: Optional name for the queue operation.
    """
    dtypes = _as_type_list(dtypes)
    shapes = _as_shape_list(shapes, dtypes)
    names = _as_name_list(names, dtypes)
    seed1, seed2 = random_seed.get_seed(seed)
    if seed1 is None and seed2 is None:
      seed1, seed2 = 0, 0
    elif seed is None and shared_name is not None:
      # This means that graph seed is provided but op seed is not provided.
      # If shared_name is also provided, make seed2 depend only on the graph
      # seed and shared_name. (seed2 from get_seed() is generally dependent on
      # the id of the last op created.)
      string = (str(seed1) + shared_name).encode("utf-8")
      seed2 = int(hashlib.md5(string).hexdigest()[:8], 16) & 0x7FFFFFFF
    queue_ref = gen_data_flow_ops.random_shuffle_queue_v2(
        component_types=dtypes,
        shapes=shapes,
        capacity=capacity,
        min_after_dequeue=min_after_dequeue,
        seed=seed1,
        seed2=seed2,
        shared_name=_shared_name(shared_name),
        name=name)
    super(RandomShuffleQueue, self).__init__(dtypes, shapes, names, queue_ref)
