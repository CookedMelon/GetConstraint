@tf_export("test.Benchmark")
class TensorFlowBenchmark(Benchmark):
  """Abstract class that provides helpers for TensorFlow benchmarks."""
  def __init__(self):
    # Allow TensorFlow runtime to allocate a new threadpool with different
    # number of threads for each new benchmark.
    os.environ[OVERRIDE_GLOBAL_THREADPOOL] = "1"
    super().__init__()
  @classmethod
  def is_abstract(cls):
    # mro: (_BenchmarkRegistrar, Benchmark, TensorFlowBenchmark) means
    # this is TensorFlowBenchmark.
    return len(cls.mro()) <= 3
  def run_op_benchmark(self,
                       sess,
                       op_or_tensor,
                       feed_dict=None,
                       burn_iters=2,
                       min_iters=10,
                       store_trace=False,
                       store_memory_usage=True,
                       name=None,
                       extras=None,
                       mbs=0):
    """Run an op or tensor in the given session.  Report the results.
    Args:
      sess: `Session` object to use for timing.
      op_or_tensor: `Operation` or `Tensor` to benchmark.
      feed_dict: A `dict` of values to feed for each op iteration (see the
        `feed_dict` parameter of `Session.run`).
      burn_iters: Number of burn-in iterations to run.
      min_iters: Minimum number of iterations to use for timing.
      store_trace: Boolean, whether to run an extra untimed iteration and
        store the trace of iteration in returned extras.
        The trace will be stored as a string in Google Chrome trace format
        in the extras field "full_trace_chrome_format". Note that trace
        will not be stored in test_log_pb2.TestResults proto.
      store_memory_usage: Boolean, whether to run an extra untimed iteration,
        calculate memory usage, and store that in extras fields.
      name: (optional) Override the BenchmarkEntry name with `name`.
        Otherwise it is inferred from the top-level method name.
      extras: (optional) Dict mapping string keys to additional benchmark info.
        Values may be either floats or values that are convertible to strings.
      mbs: (optional) The number of megabytes moved by this op, used to
        calculate the ops throughput.
    Returns:
      A `dict` containing the key-value pairs that were passed to
      `report_benchmark`. If `store_trace` option is used, then
      `full_chrome_trace_format` will be included in return dictionary even
      though it is not passed to `report_benchmark` with `extras`.
    """
    for _ in range(burn_iters):
      sess.run(op_or_tensor, feed_dict=feed_dict)
    deltas = [None] * min_iters
    for i in range(min_iters):
      start_time = time.time()
      sess.run(op_or_tensor, feed_dict=feed_dict)
      end_time = time.time()
      delta = end_time - start_time
      deltas[i] = delta
    extras = extras if extras is not None else {}
    unreported_extras = {}
    if store_trace or store_memory_usage:
      run_options = config_pb2.RunOptions(
          trace_level=config_pb2.RunOptions.FULL_TRACE)
      run_metadata = config_pb2.RunMetadata()
      sess.run(op_or_tensor, feed_dict=feed_dict,
               options=run_options, run_metadata=run_metadata)
      tl = timeline.Timeline(run_metadata.step_stats)
      if store_trace:
        unreported_extras["full_trace_chrome_format"] = (
            tl.generate_chrome_trace_format())
      if store_memory_usage:
        step_stats_analysis = tl.analyze_step_stats(show_memory=True)
        allocator_maximums = step_stats_analysis.allocator_maximums
        for k, v in allocator_maximums.items():
          extras["allocator_maximum_num_bytes_%s" % k] = v.num_bytes
    def _median(x):
      if not x:
        return -1
      s = sorted(x)
      l = len(x)
      lm1 = l - 1
      return (s[l//2] + s[lm1//2]) / 2.0
    def _mean_and_stdev(x):
      if not x:
        return -1, -1
      l = len(x)
      mean = sum(x) / l
      if l == 1:
        return mean, -1
      variance = sum([(e - mean) * (e - mean) for e in x]) / (l - 1)
      return mean, math.sqrt(variance)
    median_delta = _median(deltas)
    benchmark_values = {
        "iters": min_iters,
        "wall_time": median_delta,
        "extras": extras,
        "name": name,
        "throughput": mbs / median_delta
    }
    self.report_benchmark(**benchmark_values)
    mean_delta, stdev_delta = _mean_and_stdev(deltas)
    unreported_extras["wall_time_mean"] = mean_delta
    unreported_extras["wall_time_stdev"] = stdev_delta
    benchmark_values["extras"].update(unreported_extras)
    return benchmark_values
  def evaluate(self, tensors):
    """Evaluates tensors and returns numpy values.
    Args:
      tensors: A Tensor or a nested list/tuple of Tensors.
    Returns:
      tensors numpy values.
    """
    sess = ops.get_default_session() or self.cached_session()
    return sess.run(tensors)
