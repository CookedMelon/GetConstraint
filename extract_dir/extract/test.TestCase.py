@tf_export("test.TestCase")
class TensorFlowTestCase(googletest.TestCase):
  """Base class for tests that need to test TensorFlow."""
  def __init__(self, methodName="runTest"):  # pylint: disable=invalid-name
    super().__init__(methodName)
    # Make sure we get unfiltered stack traces during the test
    traceback_utils.disable_traceback_filtering()
    if is_xla_enabled():
      pywrap_tf_session.TF_SetXlaAutoJitMode("2")
      pywrap_tf_session.TF_SetXlaMinClusterSize(1)
      pywrap_tf_session.TF_SetXlaEnableLazyCompilation(False)
      pywrap_tf_session.TF_SetTfXlaCpuGlobalJit(True)
      # Constant folding secretly runs code on TF:Classic CPU, so we also
      # disable it here.
      pywrap_tf_session.TF_SetXlaConstantFoldingDisabled(True)
    # Check if the mlir bridge has been explicitly enabled or disabled. If
    # is_mlir_bridge_enabled() returns None, the user did not explictly enable
    # or disable the bridge so do not update enable_mlir_bridge.
    if is_mlir_bridge_enabled():
      context.context().enable_mlir_bridge = True
    elif is_mlir_bridge_enabled() is not None:
      context.context().enable_mlir_bridge = False
    self._threads = []
    self._tempdir = None
    self._cached_session = None
    self._test_start_time = None
    # This flag provides the ability to control whether the graph mode gets
    # initialized for TF1 or not. Initializing for TF1, which is what was
    # happening earlier, was preventing enablement of 'eager mode' in the test.
    self._set_default_seed = True
  def setUp(self):
    super().setUp()
    self._ClearCachedSession()
    random.seed(random_seed.DEFAULT_GRAPH_SEED)
    np.random.seed(random_seed.DEFAULT_GRAPH_SEED)
    # Note: The following line is necessary because some test methods may error
    # out from within nested graph contexts (e.g., via assertRaises and
    # assertRaisesRegexp), which may leave ops._default_graph_stack non-empty
    # under certain versions of Python. That would cause
    # ops.reset_default_graph() to throw an exception if the stack were not
    # cleared first.
    ops._default_graph_stack.reset()  # pylint: disable=protected-access
    ops.reset_default_graph()
    if self._set_default_seed:
      random_seed.set_random_seed(random_seed.DEFAULT_GRAPH_SEED)
    # Reset summary writer in case another test used set_as_default() with their
    # summary writer.
    summary_state = summary_ops_v2._summary_state  # pylint: disable=protected-access
    summary_state.writer = None
    # Avoiding calling setUp() for the poorly named test_session method.
    if self.id().endswith(".test_session"):
      self.skipTest("Not a test.")
    self._test_start_time = time.time()
  def tearDown(self):
    # If a subclass overrides setUp and doesn't call the parent class's setUp,
    # then we may not have set the start time.
    if self._test_start_time is not None:
      logging.info("time(%s): %ss", self.id(),
                   round(time.time() - self._test_start_time, 2))
    for thread in self._threads:
      thread.check_termination()
    self._ClearCachedSession()
    super().tearDown()
  def _ClearCachedSession(self):
    if self._cached_session is not None:
      self._cached_session.close()
      self._cached_session = None
  def get_temp_dir(self):
    """Returns a unique temporary directory for the test to use.
    If you call this method multiple times during in a test, it will return the
    same folder. However, across different runs the directories will be
    different. This will ensure that across different runs tests will not be
    able to pollute each others environment.
    If you need multiple unique directories within a single test, you should
    use tempfile.mkdtemp as follows:
      tempfile.mkdtemp(dir=self.get_temp_dir()):
    Returns:
      string, the path to the unique temporary directory created for this test.
    """
    if not self._tempdir:
      self._tempdir = tempfile.mkdtemp(dir=googletest.GetTempDir())
    return self._tempdir
  @contextlib.contextmanager
  def captureWritesToStream(self, stream):
    """A context manager that captures the writes to a given stream.
    This context manager captures all writes to a given stream inside of a
    `CapturedWrites` object. When this context manager is created, it yields
    the `CapturedWrites` object. The captured contents can be accessed  by
    calling `.contents()` on the `CapturedWrites`.
    For this function to work, the stream must have a file descriptor that
    can be modified using `os.dup` and `os.dup2`, and the stream must support
    a `.flush()` method. The default python sys.stdout and sys.stderr are
    examples of this. Note that this does not work in Colab or Jupyter
    notebooks, because those use alternate stdout streams.
    Example:
    ```python
    class MyOperatorTest(test_util.TensorFlowTestCase):
      def testMyOperator(self):
        input = [1.0, 2.0, 3.0, 4.0, 5.0]
        with self.captureWritesToStream(sys.stdout) as captured:
          result = MyOperator(input).eval()
        self.assertStartsWith(captured.contents(), "This was printed.")
    ```
    Args:
      stream: The stream whose writes should be captured. This stream must have
        a file descriptor, support writing via using that file descriptor, and
        must have a `.flush()` method.
    Yields:
      A `CapturedWrites` object that contains all writes to the specified stream
      made during this context.
    """
    stream.flush()
    fd = stream.fileno()
    tmp_file, tmp_file_path = tempfile.mkstemp(dir=self.get_temp_dir())
    orig_fd = os.dup(fd)
    os.dup2(tmp_file, fd)
    try:
      yield CapturedWrites(tmp_file_path)
    finally:
      os.close(tmp_file)
      os.dup2(orig_fd, fd)
  def _AssertProtoEquals(self, a, b, msg=None, relative_tolerance=None):
    """Asserts that a and b are the same proto.
    Uses ProtoEq() first, as it returns correct results
    for floating point attributes, and then use assertProtoEqual()
    in case of failure as it provides good error messages.
    Args:
      a: a proto.
      b: another proto.
      msg: Optional message to report on failure.
      relative_tolerance: float. The allowable difference between the two values
        being compared is determined by multiplying the relative tolerance by
        the maximum of the two values. If this is not provided, then all floats
        are compared using string comparison.
    """
    if not compare.ProtoEq(a, b):
      compare.assertProtoEqual(
          self,
          a,
          b,
          normalize_numbers=True,
          msg=msg,
          relative_tolerance=relative_tolerance,
      )
  def assertProtoEquals(
      self,
      expected_message_maybe_ascii,
      message,
      msg=None,
      relative_tolerance=None,
  ):
    """Asserts that message is same as parsed expected_message_ascii.
    Creates another prototype of message, reads the ascii message into it and
    then compares them using self._AssertProtoEqual().
    Args:
      expected_message_maybe_ascii: proto message in original or ascii form.
      message: the message to validate.
      msg: Optional message to report on failure.
      relative_tolerance: float. The allowable difference between the two values
        being compared is determined by multiplying the relative tolerance by
        the maximum of the two values. If this is not provided, then all floats
        are compared using string comparison.
    """
    if isinstance(expected_message_maybe_ascii, type(message)):
      expected_message = expected_message_maybe_ascii
      self._AssertProtoEquals(
          expected_message,
          message,
          msg=msg,
          relative_tolerance=relative_tolerance,
      )
    elif isinstance(expected_message_maybe_ascii, (str, bytes)):
      expected_message = type(message)()
      text_format.Merge(
          expected_message_maybe_ascii,
          expected_message,
          descriptor_pool=descriptor_pool.Default())
      self._AssertProtoEquals(
          expected_message,
          message,
          msg=msg,
          relative_tolerance=relative_tolerance,
      )
    else:
      assert False, ("Can't compare protos of type %s and %s." %
                     (type(expected_message_maybe_ascii), type(message)))
  def assertProtoEqualsVersion(
      self,
      expected,
      actual,
      producer=versions.GRAPH_DEF_VERSION,
      min_consumer=versions.GRAPH_DEF_VERSION_MIN_CONSUMER,
      msg=None):
    expected = "versions { producer: %d min_consumer: %d };\n%s" % (
        producer, min_consumer, expected)
    self.assertProtoEquals(expected, actual, msg=msg)
  def assertStartsWith(self, actual, expected_start, msg=None):
    """Assert that actual.startswith(expected_start) is True.
    Args:
      actual: str
      expected_start: str
      msg: Optional message to report on failure.
    """
    if not actual.startswith(expected_start):
      fail_msg = "%r does not start with %r" % (actual, expected_start)
      fail_msg += " : %r" % (msg) if msg else ""
      self.fail(fail_msg)
  def _eval_tensor(self, tensor):
    if tensor is None:
      return None
    elif callable(tensor):
      return self._eval_helper(tensor())
    else:
      try:
        # for compatibility with TF1 test cases
        if sparse_tensor.is_sparse(tensor):
          return sparse_tensor.SparseTensorValue(tensor.indices.numpy(),
                                                 tensor.values.numpy(),
                                                 tensor.dense_shape.numpy())
        elif ragged_tensor.is_ragged(tensor):
          return ragged_tensor_value.RaggedTensorValue(
              self._eval_tensor(tensor.values),
              self._eval_tensor(tensor.row_splits))
        elif isinstance(tensor, indexed_slices.IndexedSlices):
          return indexed_slices.IndexedSlicesValue(
              values=tensor.values.numpy(),
              indices=tensor.indices.numpy(),
              dense_shape=None
              if tensor.dense_shape is None else tensor.dense_shape.numpy())
        else:
          if hasattr(tensor, "numpy") and callable(tensor.numpy):
            return tensor.numpy()
          else:
            # Try our best to convert CompositeTensor components to NumPy
            # arrays. Officially, we don't support NumPy arrays as
            # CompositeTensor components. So don't be surprised if this doesn't
            # work.
            return nest.map_structure(lambda t: t.numpy(), tensor,
                                      expand_composites=True)
      except AttributeError as e:
        raise ValueError(f"Unsupported type {type(tensor).__name__!r}.") from e
  def _eval_helper(self, tensors):
    if tensors is None:
      return None
    return nest.map_structure(self._eval_tensor, tensors)
  def evaluate(self, tensors):
    """Evaluates tensors and returns numpy values.
    Args:
      tensors: A Tensor or a nested list/tuple of Tensors.
    Returns:
      tensors numpy values.
    """
    if context.executing_eagerly():
      return self._eval_helper(tensors)
    else:
      sess = ops.get_default_session()
      if sess is None:
        with self.test_session() as sess:
          return sess.run(tensors)
      else:
        return sess.run(tensors)
  # pylint: disable=g-doc-return-or-yield
  @contextlib.contextmanager
  def session(self, graph=None, config=None, use_gpu=True, force_gpu=False):
    """A context manager for a TensorFlow Session for use in executing tests.
    Note that this will set this session and the graph as global defaults.
    Use the `use_gpu` and `force_gpu` options to control where ops are run. If
    `force_gpu` is True, all ops are pinned to `/device:GPU:0`. Otherwise, if
    `use_gpu` is True, TensorFlow tries to run as many ops on the GPU as
    possible. If both `force_gpu and `use_gpu` are False, all ops are pinned to
    the CPU.
    Example:
    ``` python
    class MyOperatorTest(test_util.TensorFlowTestCase):
      def testMyOperator(self):
        with self.session():
          valid_input = [1.0, 2.0, 3.0, 4.0, 5.0]
          result = MyOperator(valid_input).eval()
          self.assertEqual(result, [1.0, 2.0, 3.0, 5.0, 8.0]
          invalid_input = [-1.0, 2.0, 7.0]
          with self.assertRaisesOpError("negative input not supported"):
            MyOperator(invalid_input).eval()
    ```
    Args:
      graph: Optional graph to use during the returned session.
      config: An optional config_pb2.ConfigProto to use to configure the
        session.
      use_gpu: If True, attempt to run as many ops as possible on GPU.
      force_gpu: If True, pin all ops to `/device:GPU:0`.
    Yields:
      A Session object that should be used as a context manager to surround
      the graph building and execution code in a test case.
    """
    if context.executing_eagerly():
      yield EagerSessionWarner()
    else:
      with self._create_session(graph, config, force_gpu) as sess:
        with self._constrain_devices_and_set_default(sess, use_gpu, force_gpu):
          yield sess
  @contextlib.contextmanager
  def cached_session(self,
                     graph=None,
                     config=None,
                     use_gpu=True,
                     force_gpu=False):
    """Returns a TensorFlow Session for use in executing tests.
    This method behaves differently than self.session(): for performance reasons
    `cached_session` will by default reuse the same session within the same
    test. The session returned by this function will only be closed at the end
    of the test (in the TearDown function).
    Use the `use_gpu` and `force_gpu` options to control where ops are run. If
    `force_gpu` is True, all ops are pinned to `/device:GPU:0`. Otherwise, if
    `use_gpu` is True, TensorFlow tries to run as many ops on the GPU as
    possible. If both `force_gpu and `use_gpu` are False, all ops are pinned to
    the CPU.
    Example:
    ```python
    class MyOperatorTest(test_util.TensorFlowTestCase):
      def testMyOperator(self):
        with self.cached_session() as sess:
          valid_input = [1.0, 2.0, 3.0, 4.0, 5.0]
          result = MyOperator(valid_input).eval()
          self.assertEqual(result, [1.0, 2.0, 3.0, 5.0, 8.0]
          invalid_input = [-1.0, 2.0, 7.0]
          with self.assertRaisesOpError("negative input not supported"):
            MyOperator(invalid_input).eval()
    ```
    Args:
      graph: Optional graph to use during the returned session.
      config: An optional config_pb2.ConfigProto to use to configure the
        session.
      use_gpu: If True, attempt to run as many ops as possible on GPU.
      force_gpu: If True, pin all ops to `/device:GPU:0`.
    Yields:
      A Session object that should be used as a context manager to surround
      the graph building and execution code in a test case.
    """
    if context.executing_eagerly():
      yield FakeEagerSession(self)
    else:
      sess = self._get_cached_session(
          graph, config, force_gpu, crash_if_inconsistent_args=True)
      with self._constrain_devices_and_set_default(sess, use_gpu,
                                                   force_gpu) as cached:
        yield cached
  @contextlib.contextmanager
  @deprecation.deprecated(None, "Use `self.session()` or "
                          "`self.cached_session()` instead.")
  def test_session(self,
                   graph=None,
                   config=None,
                   use_gpu=True,
                   force_gpu=False):
    """Use cached_session instead."""
    if self.id().endswith(".test_session"):
      self.skipTest(
          "Tests that have the name \"test_session\" are automatically skipped "
          "by TensorFlow test fixture, as the name is reserved for creating "
          "sessions within tests. Please rename your test if you have a test "
          "with this name.")
    if context.executing_eagerly():
      yield None
    else:
      if graph is None:
        sess = self._get_cached_session(
            graph, config, force_gpu, crash_if_inconsistent_args=False)
        with self._constrain_devices_and_set_default(sess, use_gpu,
                                                     force_gpu) as cached:
          yield cached
      else:
        with self.session(graph, config, use_gpu, force_gpu) as sess:
          yield sess
  # pylint: enable=g-doc-return-or-yield
  class _CheckedThread(object):
    """A wrapper class for Thread that asserts successful completion.
    This class should be created using the TensorFlowTestCase.checkedThread()
    method.
    """
    def __init__(self, testcase, target, args=None, kwargs=None):
      """Constructs a new instance of _CheckedThread.
      Args:
        testcase: The TensorFlowTestCase for which this thread is being created.
        target: A callable object representing the code to be executed in the
          thread.
        args: A tuple of positional arguments that will be passed to target.
        kwargs: A dictionary of keyword arguments that will be passed to target.
      """
      self._testcase = testcase
      self._target = target
      self._args = () if args is None else args
      self._kwargs = {} if kwargs is None else kwargs
      self._thread = threading.Thread(target=self._protected_run)
      self._exception = None
      self._is_thread_joined = False
    def _protected_run(self):
      """Target for the wrapper thread. Sets self._exception on failure."""
      try:
        self._target(*self._args, **self._kwargs)
      except Exception as e:  # pylint: disable=broad-except
        self._exception = e
    def start(self):
      """Starts the thread's activity.
      This must be called at most once per _CheckedThread object. It arranges
      for the object's target to be invoked in a separate thread of control.
      """
      self._thread.start()
    def join(self):
      """Blocks until the thread terminates.
      Raises:
        self._testcase.failureException: If the thread terminates with due to
          an exception.
      """
      self._is_thread_joined = True
      self._thread.join()
      if self._exception is not None:
        self._testcase.fail("Error in checkedThread: %s" % str(self._exception))
    def is_alive(self):
      """Returns whether the thread is alive.
      This method returns True just before the run() method starts
      until just after the run() method terminates.
      Returns:
        True if the thread is alive, otherwise False.
      """
      return self._thread.is_alive()
    def check_termination(self):
      """Returns whether the checked thread was properly used and did terminate.
      Every checked thread should be "join"ed after starting, and before the
      test tears down. If it is not joined, it is possible the thread will hang
      and cause flaky failures in tests.
      Raises:
        self._testcase.failureException: If check_termination was called before
        thread was joined.
        RuntimeError: If the thread is not terminated. This means thread was not
        joined with the main thread.
      """
      if self._is_thread_joined:
        if self.is_alive():
          raise RuntimeError(
              "Thread was not joined with main thread, and is still running "
              "when the test finished.")
      else:
        self._testcase.fail("A checked thread was not joined.")
  def checkedThread(self, target, args=None, kwargs=None):
    """Returns a Thread wrapper that asserts 'target' completes successfully.
    This method should be used to create all threads in test cases, as
    otherwise there is a risk that a thread will silently fail, and/or
    assertions made in the thread will not be respected.
    Args:
      target: A callable object to be executed in the thread.
      args: The argument tuple for the target invocation. Defaults to ().
      kwargs: A dictionary of keyword arguments for the target invocation.
        Defaults to {}.
    Returns:
      A wrapper for threading.Thread that supports start() and join() methods.
    """
    ret = TensorFlowTestCase._CheckedThread(self, target, args, kwargs)
    self._threads.append(ret)
    return ret
  # pylint: enable=invalid-name
  @py_func_if_in_function
  def assertNear(self, f1, f2, err, msg=None):
    """Asserts that two floats are near each other.
    Checks that |f1 - f2| < err and asserts a test failure
    if not.
    Args:
      f1: A float value.
      f2: A float value.
      err: A float value.
      msg: An optional string message to append to the failure message.
    """
    # f1 == f2 is needed here as we might have: f1, f2 = inf, inf
    self.assertTrue(
        f1 == f2 or math.fabs(f1 - f2) <= err, "%f != %f +/- %f%s" %
        (f1, f2, err, " (%s)" % msg if msg is not None else ""))
  @py_func_if_in_function
  def assertArrayNear(self, farray1, farray2, err, msg=None):
    """Asserts that two float arrays are near each other.
    Checks that for all elements of farray1 and farray2
    |f1 - f2| < err.  Asserts a test failure if not.
    Args:
      farray1: a list of float values.
      farray2: a list of float values.
      err: a float value.
      msg: Optional message to report on failure.
    """
    self.assertEqual(len(farray1), len(farray2), msg=msg)
    for f1, f2 in zip(farray1, farray2):
      self.assertNear(float(f1), float(f2), err, msg=msg)
  def _NDArrayNear(self, ndarray1, ndarray2, err):
    return np.linalg.norm(ndarray1 - ndarray2) < err
  @py_func_if_in_function
  def assertNDArrayNear(self, ndarray1, ndarray2, err, msg=None):
    """Asserts that two numpy arrays have near values.
    Args:
      ndarray1: a numpy ndarray.
      ndarray2: a numpy ndarray.
      err: a float. The maximum absolute difference allowed.
      msg: Optional message to report on failure.
    """
    self.assertTrue(self._NDArrayNear(ndarray1, ndarray2, err), msg=msg)
  def _GetNdArray(self, a):
    # If a is tensor-like then convert it to ndarray
    if tensor_util.is_tf_type(a):
      if isinstance(a, ops._EagerTensorBase):
        a = a.numpy()
      else:
        a = self.evaluate(a)
    if not isinstance(a, np.ndarray):
      try:
        return np.array(a)
      except ValueError as e:
        # TODO(b/264461299): NumPy 1.24 no longer infers dtype=object from
        # ragged sequences.
        # See:
        # https://numpy.org/neps/nep-0034-infer-dtype-is-object.html
        # Fixing this correctly requires clarifying the API contract of this
        # function with respect to ragged sequences and possibly updating all
        # users. As a backwards compatibility measure, if array
        # creation fails with an "inhomogeneous shape" error, try again with
        # an explicit dtype=object, which should restore the previous behavior.
        if "inhomogeneous shape" in str(e):
          return np.array(a, dtype=object)
        else:
          raise
    return a
  def evaluate_if_both_tensors(self, a, b):
    if (tensor_util.is_tf_type(a) and tensor_util.is_tf_type(b) and
        not isinstance(a, ops._EagerTensorBase) and
        not isinstance(b, ops._EagerTensorBase)):
      return self.evaluate((a, b))
    else:
      return (a, b)
  def _assertArrayLikeAllClose(self, a, b, rtol=1e-6, atol=1e-6, msg=None):
    (a, b) = self.evaluate_if_both_tensors(a, b)
    a = self._GetNdArray(a)
    b = self._GetNdArray(b)
    # When the array rank is small, print its contents. Numpy array printing is
    # implemented using inefficient recursion so prints can cause tests to
    # time out.
    if a.shape != b.shape and (b.ndim <= 3 or b.size < 500):
      shape_mismatch_msg = ("Shape mismatch: expected %s, got %s with contents "
                            "%s.") % (a.shape, b.shape, b)
    else:
      shape_mismatch_msg = "Shape mismatch: expected %s, got %s." % (a.shape,
                                                                     b.shape)
    self.assertEqual(a.shape, b.shape, shape_mismatch_msg)
    msgs = [msg]
    # np.allclose does not always work for our custom bfloat16 and float8
    # extension types when type promotions are involved, so we first cast any
    # arrays of such types to float32.
    a_dtype = a.dtype
    custom_dtypes = (dtypes.bfloat16.as_numpy_dtype,
                     dtypes.float8_e5m2.as_numpy_dtype,
                     dtypes.float8_e4m3fn.as_numpy_dtype)
    a = a.astype(np.float32) if a.dtype in custom_dtypes else a
    b = b.astype(np.float32) if b.dtype in custom_dtypes else b
    if not np.allclose(a, b, rtol=rtol, atol=atol):
      # Adds more details to np.testing.assert_allclose.
      #
      # NOTE: numpy.allclose (and numpy.testing.assert_allclose)
      # checks whether two arrays are element-wise equal within a
      # tolerance. The relative difference (rtol * abs(b)) and the
      # absolute difference atol are added together to compare against
      # the absolute difference between a and b.  Here, we want to
      # tell user which elements violate such conditions.
      cond = np.logical_or(
          np.abs(a - b) > atol + rtol * np.abs(b),
          np.isnan(a) != np.isnan(b))
      if a.ndim:
        x = a[np.where(cond)]
        y = b[np.where(cond)]
        msgs.append("not close where = {}".format(np.where(cond)))
      else:
        # np.where is broken for scalars
        x, y = a, b
      msgs.append("not close lhs = {}".format(x))
      msgs.append("not close rhs = {}".format(y))
      msgs.append("not close dif = {}".format(np.abs(x - y)))
      msgs.append("not close tol = {}".format(atol + rtol * np.abs(y)))
      msgs.append("dtype = {}, shape = {}".format(a_dtype, a.shape))
      # TODO(xpan): There seems to be a bug:
      # tensorflow/compiler/tests:binary_ops_test pass with float32
      # nan even though the equal_nan is False by default internally.
      np.testing.assert_allclose(
          a, b, rtol=rtol, atol=atol, err_msg="\n".join(msgs), equal_nan=True)
  def _assertAllCloseRecursive(self,
                               a,
                               b,
                               rtol=1e-6,
                               atol=1e-6,
                               path=None,
                               msg=None):
    if ragged_tensor.is_ragged(a) or ragged_tensor.is_ragged(b):
      return self._assertRaggedClose(a, b, rtol, atol, msg)
    path = path or []
    path_str = (("[" + "][".join(str(p) for p in path) + "]") if path else "")
    msg = msg if msg else ""
    # Check if a and/or b are namedtuples.
    if hasattr(a, "_asdict"):
      a = a._asdict()
    if hasattr(b, "_asdict"):
      b = b._asdict()
    a_is_dict = isinstance(a, collections_abc.Mapping)
    if a_is_dict != isinstance(b, collections_abc.Mapping):
      raise ValueError("Can't compare dict to non-dict, a%s vs b%s. %s" %
                       (path_str, path_str, msg))
    if a_is_dict:
      self.assertItemsEqual(
          a.keys(),
          b.keys(),
          msg="mismatched keys: a%s has keys %s, but b%s has keys %s. %s" %
          (path_str, a.keys(), path_str, b.keys(), msg))
      for k in a:
        path.append(k)
        self._assertAllCloseRecursive(
            a[k], b[k], rtol=rtol, atol=atol, path=path, msg=msg)
        del path[-1]
    elif isinstance(a, (list, tuple)):
      # Try to directly compare a, b as ndarrays; if not work, then traverse
      # through the sequence, which is more expensive.
      try:
        (a, b) = self.evaluate_if_both_tensors(a, b)
        a_as_ndarray = self._GetNdArray(a)
        b_as_ndarray = self._GetNdArray(b)
        self._assertArrayLikeAllClose(
            a_as_ndarray,
            b_as_ndarray,
            rtol=rtol,
            atol=atol,
            msg="Mismatched value: a%s is different from b%s. %s" %
            (path_str, path_str, msg))
      except (ValueError, TypeError, NotImplementedError) as e:
        if len(a) != len(b):
          raise ValueError(
              "Mismatched length: a%s has %d items, but b%s has %d items. %s" %
              (path_str, len(a), path_str, len(b), msg))
        for idx, (a_ele, b_ele) in enumerate(zip(a, b)):
          path.append(str(idx))
          self._assertAllCloseRecursive(
              a_ele, b_ele, rtol=rtol, atol=atol, path=path, msg=msg)
          del path[-1]
    # a and b are ndarray like objects
    else:
      try:
        self._assertArrayLikeAllClose(
            a,
            b,
            rtol=rtol,
            atol=atol,
            msg=("Mismatched value: a%s is different from b%s. %s" %
                 (path_str, path_str, msg)))
      except TypeError as e:
        msg = ("Error: a%s has %s, but b%s has %s. %s" %
               (path_str, type(a), path_str, type(b), msg))
        e.args = ((e.args[0] + " : " + msg,) + e.args[1:])
        raise
  @py_func_if_in_function
  def assertAllClose(self, a, b, rtol=1e-6, atol=1e-6, msg=None):
    """Asserts that two structures of numpy arrays or Tensors, have near values.
    `a` and `b` can be arbitrarily nested structures. A layer of a nested
    structure can be a `dict`, `namedtuple`, `tuple` or `list`.
    Note: the implementation follows
    [`numpy.allclose`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.allclose.html)
    (and numpy.testing.assert_allclose). It checks whether two arrays are
    element-wise equal within a tolerance. The relative difference
    (`rtol * abs(b)`) and the absolute difference `atol` are added together
    to compare against the absolute difference between `a` and `b`.
    Args:
      a: The expected numpy `ndarray`, or anything that can be converted into a
        numpy `ndarray` (including Tensor), or any arbitrarily nested of
        structure of these.
      b: The actual numpy `ndarray`, or anything that can be converted into a
        numpy `ndarray` (including Tensor), or any arbitrarily nested of
        structure of these.
      rtol: relative tolerance.
      atol: absolute tolerance.
      msg: Optional message to report on failure.
    Raises:
      ValueError: if only one of `a[p]` and `b[p]` is a dict or
          `a[p]` and `b[p]` have different length, where `[p]` denotes a path
          to the nested structure, e.g. given `a = [(1, 1), {'d': (6, 7)}]` and
          `[p] = [1]['d']`, then `a[p] = (6, 7)`.
    """
    self._assertAllCloseRecursive(a, b, rtol=rtol, atol=atol, msg=msg)
  @py_func_if_in_function
  def assertAllCloseAccordingToType(self,
                                    a,
                                    b,
                                    rtol=1e-6,
                                    atol=1e-6,
                                    float_rtol=1e-6,
                                    float_atol=1e-6,
                                    half_rtol=1e-3,
                                    half_atol=1e-3,
                                    bfloat16_rtol=1e-2,
                                    bfloat16_atol=1e-2,
                                    msg=None):
    """Like assertAllClose, but also suitable for comparing fp16 arrays.
    In particular, the tolerance is reduced to 1e-3 if at least
    one of the arguments is of type float16.
    Args:
      a: the expected numpy ndarray or anything can be converted to one.
      b: the actual numpy ndarray or anything can be converted to one.
      rtol: relative tolerance.
      atol: absolute tolerance.
      float_rtol: relative tolerance for float32.
      float_atol: absolute tolerance for float32.
      half_rtol: relative tolerance for float16.
      half_atol: absolute tolerance for float16.
      bfloat16_rtol: relative tolerance for bfloat16.
      bfloat16_atol: absolute tolerance for bfloat16.
      msg: Optional message to report on failure.
    """
    (a, b) = self.evaluate_if_both_tensors(a, b)
    a = self._GetNdArray(a)
    b = self._GetNdArray(b)
    # types with lower tol are put later to overwrite previous ones.
    if (a.dtype == np.float32 or b.dtype == np.float32 or
        a.dtype == np.complex64 or b.dtype == np.complex64):
      rtol = max(rtol, float_rtol)
      atol = max(atol, float_atol)
    if a.dtype == np.float16 or b.dtype == np.float16:
      rtol = max(rtol, half_rtol)
      atol = max(atol, half_atol)
    if (a.dtype == dtypes.bfloat16.as_numpy_dtype or
        b.dtype == dtypes.bfloat16.as_numpy_dtype):
      rtol = max(rtol, bfloat16_rtol)
      atol = max(atol, bfloat16_atol)
    self.assertAllClose(a, b, rtol=rtol, atol=atol, msg=msg)
  @py_func_if_in_function
  def assertNotAllClose(self, a, b, rtol=1e-6, atol=1e-6, msg=None):
    """Assert that two numpy arrays, or Tensors, do not have near values.
    Args:
      a: The expected numpy `ndarray`, or anything that can be converted into a
        numpy `ndarray` (including Tensor), or any arbitrarily nested of
        structure of these.
      b: The actual numpy `ndarray`, or anything that can be converted into a
        numpy `ndarray` (including Tensor), or any arbitrarily nested of
        structure of these.
      rtol: relative tolerance.
      atol: absolute tolerance.
      msg: Optional message to report on failure.
    Raises:
      AssertionError: If `a` and `b` are unexpectedly close at all elements.
    """
    try:
      self.assertAllClose(a, b, rtol=rtol, atol=atol, msg=msg)
    except AssertionError:
      return
    msg = msg or ""
    raise AssertionError("The two values are close at all elements. %s" % msg)
  @py_func_if_in_function
  def assertAllEqual(self, a, b, msg=None):
    """Asserts that two numpy arrays or Tensors have the same values.
    Args:
      a: the expected numpy ndarray or anything can be converted to one.
      b: the actual numpy ndarray or anything can be converted to one.
      msg: Optional message to report on failure.
    """
    if (ragged_tensor.is_ragged(a) or ragged_tensor.is_ragged(b)):
      return self._assertRaggedEqual(a, b, msg)
    msg = msg if msg else ""
    (a, b) = self.evaluate_if_both_tensors(a, b)
    a = self._GetNdArray(a)
    b = self._GetNdArray(b)
    # Arbitrary bounds so that we don't print giant tensors.
    if (b.ndim <= 3 or b.size < 500):
      self.assertEqual(
          a.shape, b.shape, "Shape mismatch: expected %s, got %s."
          " Contents: %r. \n%s." % (a.shape, b.shape, b, msg))
    else:
      self.assertEqual(
          a.shape, b.shape, "Shape mismatch: expected %s, got %s."
          " %s" % (a.shape, b.shape, msg))
    same = (a == b)
    if dtypes.as_dtype(a.dtype).is_floating:
      same = np.logical_or(same, np.logical_and(np.isnan(a), np.isnan(b)))
    msgs = [msg]
    if not np.all(same):
      # Adds more details to np.testing.assert_array_equal.
      diff = np.logical_not(same)
      if a.ndim:
        x = a[np.where(diff)]
        y = b[np.where(diff)]
        msgs.append("not equal where = {}".format(np.where(diff)))
      else:
        # np.where is broken for scalars
        x, y = a, b
      msgs.append("not equal lhs = %r" % x)
      msgs.append("not equal rhs = %r" % y)
      if (a.dtype.kind != b.dtype.kind and
          {a.dtype.kind, b.dtype.kind}.issubset({"U", "S", "O"})):
        a_list = []
        b_list = []
        # OK to flatten `a` and `b` because they are guaranteed to have the
        # same shape.
        for out_list, flat_arr in [(a_list, a.flat), (b_list, b.flat)]:
          for item in flat_arr:
            if isinstance(item, str):
              out_list.append(item.encode("utf-8"))
            else:
              out_list.append(item)
        a = np.array(a_list)
        b = np.array(b_list)
      np.testing.assert_array_equal(a, b, err_msg="\n".join(msgs))
  @py_func_if_in_function
  def assertNotAllEqual(self, a, b, msg=None):
    """Asserts that two numpy arrays or Tensors do not have the same values.
    Args:
      a: the expected numpy ndarray or anything can be converted to one.
      b: the actual numpy ndarray or anything can be converted to one.
      msg: Optional message to report on failure.
    """
    try:
      self.assertAllEqual(a, b)
    except AssertionError:
      return
    msg = msg or ""
    raise AssertionError("The two values are equal at all elements. %s" % msg)
  @py_func_if_in_function
  def assertAllGreater(self, a, comparison_target):
    """Assert element values are all greater than a target value.
    Args:
      a: The numpy `ndarray`, or anything that can be converted into a numpy
        `ndarray` (including Tensor).
      comparison_target: The target value of comparison.
    """
    (a, comparison_target) = self.evaluate_if_both_tensors(a, comparison_target)
    a = self._GetNdArray(a)
    self.assertGreater(np.min(a), comparison_target)
  @py_func_if_in_function
  def assertAllLess(self, a, comparison_target):
    """Assert element values are all less than a target value.
    Args:
      a: The numpy `ndarray`, or anything that can be converted into a numpy
        `ndarray` (including Tensor).
      comparison_target: The target value of comparison.
    """
    (a, comparison_target) = self.evaluate_if_both_tensors(a, comparison_target)
    a = self._GetNdArray(a)
    self.assertLess(np.max(a), comparison_target)
  @py_func_if_in_function
  def assertAllGreaterEqual(self, a, comparison_target):
    """Assert element values are all greater than or equal to a target value.
    Args:
      a: The numpy `ndarray`, or anything that can be converted into a numpy
        `ndarray` (including Tensor).
      comparison_target: The target value of comparison.
    """
    (a, comparison_target) = self.evaluate_if_both_tensors(a, comparison_target)
    a = self._GetNdArray(a)
    self.assertGreaterEqual(np.min(a), comparison_target)
  @py_func_if_in_function
  def assertAllLessEqual(self, a, comparison_target):
    """Assert element values are all less than or equal to a target value.
    Args:
      a: The numpy `ndarray`, or anything that can be converted into a numpy
        `ndarray` (including Tensor).
      comparison_target: The target value of comparison.
    """
    (a, comparison_target) = self.evaluate_if_both_tensors(a, comparison_target)
    a = self._GetNdArray(a)
    self.assertLessEqual(np.max(a), comparison_target)
  def _format_subscripts(self, subscripts, value, limit=10, indent=2):
    """Generate a summary of ndarray subscripts as a list of str.
    If limit == N, this method will print up to the first N subscripts on
    separate
    lines. A line of ellipses (...) will be appended at the end if the number of
    subscripts exceeds N.
    Args:
      subscripts: The tensor (np.ndarray) subscripts, of the same format as
        np.where()'s return value, i.e., a tuple of arrays with each array
        corresponding to a dimension. E.g., (array([1, 1]), array([0, 1])).
      value: (np.ndarray) value of the tensor.
      limit: (int) The maximum number of indices to print.
      indent: (int) Number of characters to indent at the beginning of each
        line.
    Returns:
      (list of str) the multi-line representation of the subscripts and values,
        potentially with omission at the end.
    """
    lines = []
    subscripts = np.transpose(subscripts)
    prefix = " " * indent
    if np.ndim(value) == 0:
      return [prefix + "[0] : " + str(value)]
    for subscript in itertools.islice(subscripts, limit):
      lines.append(prefix + str(subscript) + " : " +
                   str(value[tuple(subscript)]))
    if len(subscripts) > limit:
      lines.append(prefix + "...")
    return lines
  @py_func_if_in_function
  def assertAllInRange(self,
                       target,
                       lower_bound,
                       upper_bound,
                       open_lower_bound=False,
                       open_upper_bound=False):
    """Assert that elements in a Tensor are all in a given range.
    Args:
      target: The numpy `ndarray`, or anything that can be converted into a
        numpy `ndarray` (including Tensor).
      lower_bound: lower bound of the range
      upper_bound: upper bound of the range
      open_lower_bound: (`bool`) whether the lower bound is open (i.e., > rather
        than the default >=)
      open_upper_bound: (`bool`) whether the upper bound is open (i.e., < rather
        than the default <=)
    Raises:
      AssertionError:
        if the value tensor does not have an ordered numeric type (float* or
          int*), or
        if there are nan values, or
        if any of the elements do not fall in the specified range.
    """
    target = self._GetNdArray(target)
    if not (np.issubdtype(target.dtype, np.floating) or
            np.issubdtype(target.dtype, np.integer)):
      raise AssertionError(
          "The value of %s does not have an ordered numeric type, instead it "
          "has type: %s" % (target, target.dtype))
    nan_subscripts = np.where(np.isnan(target))
    if np.size(nan_subscripts):
      raise AssertionError(
          "%d of the %d element(s) are NaN. "
          "Subscripts(s) and value(s) of the NaN element(s):\n" %
          (len(nan_subscripts[0]), np.size(target)) +
          "\n".join(self._format_subscripts(nan_subscripts, target)))
    range_str = (("(" if open_lower_bound else "[") + str(lower_bound) + ", " +
                 str(upper_bound) + (")" if open_upper_bound else "]"))
    violations = (
        np.less_equal(target, lower_bound) if open_lower_bound else np.less(
            target, lower_bound))
    violations = np.logical_or(
        violations,
        np.greater_equal(target, upper_bound)
        if open_upper_bound else np.greater(target, upper_bound))
    violation_subscripts = np.where(violations)
    if np.size(violation_subscripts):
      raise AssertionError(
          "%d of the %d element(s) are outside the range %s. " %
          (len(violation_subscripts[0]), np.size(target), range_str) +
          "Subscript(s) and value(s) of the offending elements:\n" +
          "\n".join(self._format_subscripts(violation_subscripts, target)))
  @py_func_if_in_function
  def assertAllInSet(self, target, expected_set):
    """Assert that elements of a Tensor are all in a given closed set.
    Args:
      target: The numpy `ndarray`, or anything that can be converted into a
        numpy `ndarray` (including Tensor).
      expected_set: (`list`, `tuple` or `set`) The closed set that the elements
        of the value of `target` are expected to fall into.
    Raises:
      AssertionError:
        if any of the elements do not fall into `expected_set`.
    """
    target = self._GetNdArray(target)
    # Elements in target that are not in expected_set.
    diff = np.setdiff1d(target.flatten(), list(expected_set))
    if np.size(diff):
      raise AssertionError("%d unique element(s) are not in the set %s: %s" %
                           (np.size(diff), expected_set, diff))
  @py_func_if_in_function
  def assertDTypeEqual(self, target, expected_dtype):
    """Assert ndarray data type is equal to expected.
    Args:
      target: The numpy `ndarray`, or anything that can be converted into a
        numpy `ndarray` (including Tensor).
      expected_dtype: Expected data type.
    """
    target = self._GetNdArray(target)
    if not isinstance(target, list):
      arrays = [target]
    for arr in arrays:
      self.assertEqual(arr.dtype, expected_dtype)
  # pylint: disable=g-doc-return-or-yield
  @contextlib.contextmanager
  def assertRaisesWithPredicateMatch(self, exception_type,
                                     expected_err_re_or_predicate):
    """Returns a context manager to enclose code expected to raise an exception.
    If the exception is an OpError, the op stack is also included in the message
    predicate search.
    Args:
      exception_type: The expected type of exception that should be raised.
      expected_err_re_or_predicate: If this is callable, it should be a function
        of one argument that inspects the passed-in exception and returns True
        (success) or False (please fail the test). Otherwise, the error message
        is expected to match this regular expression partially.
    Returns:
      A context manager to surround code that is expected to raise an
      exception.
    """
    if callable(expected_err_re_or_predicate):
      predicate = expected_err_re_or_predicate
    else:
      def predicate(e):
        err_str = e.message if isinstance(e, errors.OpError) else str(e)
        op = e.op if isinstance(e, errors.OpError) else None
        while op is not None:
          err_str += "\nCaused by: " + op.name
          op = op._original_op  # pylint: disable=protected-access
        logging.info("Searching within error strings: '%s' within '%s'",
                     expected_err_re_or_predicate, err_str)
        return re.search(expected_err_re_or_predicate, err_str)
    try:
      yield
      self.fail(exception_type.__name__ + " not raised")
    except Exception as e:  # pylint: disable=broad-except
      if not isinstance(e, exception_type) or not predicate(e):
        raise AssertionError("Exception of type %s: %s" %
                             (str(type(e)), str(e)))
  # pylint: enable=g-doc-return-or-yield
  def assertRaisesOpError(self, expected_err_re_or_predicate):
    return self.assertRaisesWithPredicateMatch(errors.OpError,
                                               expected_err_re_or_predicate)
  def assertRaisesIncompatibleShapesError(
      self, exception_type=errors.InvalidArgumentError):
    return self.assertRaisesWithPredicateMatch(
        exception_type, r"Incompatible shapes|Dimensions must be equal|"
        r"required broadcastable shapes")
  def assertShapeEqual(self, input_a, input_b, msg=None):
    """Asserts that two Numpy or TensorFlow objects have the same shape.
    For Tensors, this compares statically known shapes at compile time, not
    dynamic shapes at runtime.
    Args:
      input_a: A Numpy ndarray, Numpy scalar, or a Tensor.
      input_b: A Numpy ndarray, Numpy scalar, or a Tensor.
      msg: Optional message to report on failure.
    Raises:
      TypeError: If the arguments have the wrong type.
    """
    if not isinstance(input_a, (np.ndarray, np.generic, ops.Tensor)):
      raise TypeError(
          "input_a must be a Numpy ndarray, Numpy scalar, or a Tensor."
          f"Instead received {type(input_a)}")
    if not isinstance(input_b, (np.ndarray, np.generic, ops.Tensor)):
      raise TypeError(
          "input_b must be a Numpy ndarray, Numpy scalar, or a Tensor."
          f"Instead received {type(input_b)}")
    shape_a = input_a.get_shape().as_list() if isinstance(
        input_a, ops.Tensor) else input_a.shape
    shape_b = input_b.get_shape().as_list() if isinstance(
        input_b, ops.Tensor) else input_b.shape
    self.assertAllEqual(shape_a, shape_b, msg=msg)
  def assertDeviceEqual(self, device1, device2, msg=None):
    """Asserts that the two given devices are the same.
    Args:
      device1: A string device name or TensorFlow `DeviceSpec` object.
      device2: A string device name or TensorFlow `DeviceSpec` object.
      msg: Optional message to report on failure.
    """
    device1 = pydev.canonical_name(device1)
    device2 = pydev.canonical_name(device2)
    self.assertEqual(
        device1, device2,
        "Devices %s and %s are not equal. %s" % (device1, device2, msg))
  @py_func_if_in_function
  def assertDictEqual(self, a, b, msg=None):
    """Assert that two given dictionary of tensors are the same.
    Args:
      a: Expected dictionary with numpy ndarray or anything else that can be
        converted to one as values.
      b: Actual dictionary with numpy ndarray or anything else that can be
        converted to one as values.
      msg: Optional message to report on failure.
    """
    # To keep backwards compatibility, we first try the base class
    # assertDictEqual. If that fails we try the tensorflow one.
    try:
      super().assertDictEqual(a, b, msg)
    except Exception:  # pylint: disable=broad-except
      self.assertSameElements(a.keys(), b.keys())  # pylint: disable=g-assert-in-except
      for k, v in a.items():
        (a_k, b_k) = self.evaluate_if_both_tensors(v, b[k])
        a_k = self._GetNdArray(a_k)
        b_k = self._GetNdArray(b_k)
        if np.issubdtype(a_k.dtype, np.floating):
          self.assertAllClose(v, b[k], msg=k)
        else:
          self.assertAllEqual(v, b[k], msg=k)
  def _GetPyList(self, a):
    """Converts `a` to a nested python list."""
    if isinstance(a, ragged_tensor.RaggedTensor):
      return self.evaluate(a).to_list()
    elif isinstance(a, ops.Tensor):
      a = self.evaluate(a)
      return a.tolist() if isinstance(a, np.ndarray) else a
    elif isinstance(a, np.ndarray):
      return a.tolist()
    elif isinstance(a, ragged_tensor_value.RaggedTensorValue):
      return a.to_list()
    else:
      return np.array(a, dtype=object).tolist()
  def _assertRaggedEqual(self, a, b, msg):
    """Asserts that two ragged tensors are equal."""
    a_list = self._GetPyList(a)
    b_list = self._GetPyList(b)
    self.assertEqual(a_list, b_list, msg)
    if not (isinstance(a, (list, tuple)) or isinstance(b, (list, tuple))):
      a_ragged_rank = a.ragged_rank if ragged_tensor.is_ragged(a) else 0
      b_ragged_rank = b.ragged_rank if ragged_tensor.is_ragged(b) else 0
      self.assertEqual(a_ragged_rank, b_ragged_rank, msg)
  def _assertRaggedClose(self, a, b, rtol, atol, msg=None):
    a_list = self._GetPyList(a)
    b_list = self._GetPyList(b)
    self._assertListCloseRecursive(a_list, b_list, rtol, atol, msg)
    if not (isinstance(a, (list, tuple)) or isinstance(b, (list, tuple))):
      a_ragged_rank = a.ragged_rank if ragged_tensor.is_ragged(a) else 0
      b_ragged_rank = b.ragged_rank if ragged_tensor.is_ragged(b) else 0
      self.assertEqual(a_ragged_rank, b_ragged_rank, msg)
  def _assertListCloseRecursive(self, a, b, rtol, atol, msg, path="value"):
    self.assertEqual(type(a), type(b))
    if isinstance(a, (list, tuple)):
      self.assertLen(a, len(b), "Length differs for %s" % path)
      for i in range(len(a)):
        self._assertListCloseRecursive(a[i], b[i], rtol, atol, msg,
                                       "%s[%s]" % (path, i))
    else:
      self._assertAllCloseRecursive(a, b, rtol, atol, path, msg)
  # Fix Python 3+ compatibility issues
  # pylint: disable=invalid-name
  # Silence a deprecation warning
  assertRaisesRegexp = googletest.TestCase.assertRaisesRegex
  # assertItemsEqual is assertCountEqual as of 3.2.
  assertItemsEqual = googletest.TestCase.assertCountEqual
  # pylint: enable=invalid-name
  @contextlib.contextmanager
  def _constrain_devices_and_set_default(self, sess, use_gpu, force_gpu):
    """Set the session and its graph to global default and constrain devices."""
    if context.executing_eagerly():
      yield None
    else:
      with sess.graph.as_default(), sess.as_default():
        if force_gpu:
          # Use the name of an actual device if one is detected, or
          # '/device:GPU:0' otherwise
          gpu_name = gpu_device_name()
          if not gpu_name:
            gpu_name = "/device:GPU:0"
          with sess.graph.device(gpu_name):
            yield sess
        elif use_gpu:
          yield sess
        else:
          with sess.graph.device("/device:CPU:0"):
            yield sess
  def _create_session(self, graph, config, force_gpu):
    """See session() for details."""
    def prepare_config(config):
      """Returns a config for sessions.
      Args:
        config: An optional config_pb2.ConfigProto to use to configure the
          session.
      Returns:
        A config_pb2.ConfigProto object.
      """
      # TODO(b/114333779): Enforce allow_soft_placement=False when
      # use_gpu=False. Currently many tests rely on the fact that any device
      # will be used even when a specific device is supposed to be used.
      allow_soft_placement = not force_gpu
      if config is None:
        config = context.context().config
        config.allow_soft_placement = allow_soft_placement
      elif not allow_soft_placement and config.allow_soft_placement:
        config_copy = context.context().config
        config = config_copy
        config.allow_soft_placement = False
      # Don't perform optimizations for tests so we don't inadvertently run
      # gpu ops on cpu
      config.graph_options.optimizer_options.opt_level = -1
      # Disable Grappler constant folding since some tests & benchmarks
      # use constant input and become meaningless after constant folding.
      # DO NOT DISABLE GRAPPLER OPTIMIZERS WITHOUT CONSULTING WITH THE
      # GRAPPLER TEAM.
      config.graph_options.rewrite_options.constant_folding = (
          rewriter_config_pb2.RewriterConfig.OFF)
      config.graph_options.rewrite_options.pin_to_host_optimization = (
          rewriter_config_pb2.RewriterConfig.OFF)
      return config
    return ErrorLoggingSession(graph=graph, config=prepare_config(config))
  def _get_cached_session(self,
                          graph=None,
                          config=None,
                          force_gpu=False,
                          crash_if_inconsistent_args=True):
    """See cached_session() for documentation."""
    if self._cached_session is None:
      sess = self._create_session(
          graph=graph, config=config, force_gpu=force_gpu)
      self._cached_session = sess
      self._cached_graph = graph
      self._cached_config = config
      self._cached_force_gpu = force_gpu
      return sess
    else:
      if crash_if_inconsistent_args and self._cached_graph is not graph:
        raise ValueError("The graph used to get the cached session is "
                         "different than the one that was used to create the "
                         "session. Maybe create a new session with "
                         "self.session()")
      if crash_if_inconsistent_args and self._cached_config is not config:
        raise ValueError("The config used to get the cached session is "
                         "different than the one that was used to create the "
                         "session. Maybe create a new session with "
                         "self.session()")
      if crash_if_inconsistent_args and (self._cached_force_gpu is
                                         not force_gpu):
        raise ValueError(
            "The force_gpu value used to get the cached session is "
            "different than the one that was used to create the "
            "session. Maybe create a new session with "
            "self.session()")
      return self._cached_session
