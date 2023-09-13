@keras_export("keras.utils.SequenceEnqueuer")
class SequenceEnqueuer:
    """Base class to enqueue inputs.
    The task of an Enqueuer is to use parallelism to speed up preprocessing.
    This is done with processes or threads.
    Example:
    ```python
        enqueuer = SequenceEnqueuer(...)
        enqueuer.start()
        datas = enqueuer.get()
        for data in datas:
            # Use the inputs; training, evaluating, predicting.
            # ... stop sometime.
        enqueuer.stop()
    ```
    The `enqueuer.get()` should be an infinite stream of data.
    """
    def __init__(self, sequence, use_multiprocessing=False):
        self.sequence = sequence
        self.use_multiprocessing = use_multiprocessing
        global _SEQUENCE_COUNTER
        if _SEQUENCE_COUNTER is None:
            try:
                _SEQUENCE_COUNTER = multiprocessing.Value("i", 0)
            except OSError:
                # In this case the OS does not allow us to use
                # multiprocessing. We resort to an int
                # for enqueuer indexing.
                _SEQUENCE_COUNTER = 0
        if isinstance(_SEQUENCE_COUNTER, int):
            self.uid = _SEQUENCE_COUNTER
            _SEQUENCE_COUNTER += 1
        else:
            # Doing Multiprocessing.Value += x is not process-safe.
            with _SEQUENCE_COUNTER.get_lock():
                self.uid = _SEQUENCE_COUNTER.value
                _SEQUENCE_COUNTER.value += 1
        self.workers = 0
        self.executor_fn = None
        self.queue = None
        self.run_thread = None
        self.stop_signal = None
    def is_running(self):
        return self.stop_signal is not None and not self.stop_signal.is_set()
    def start(self, workers=1, max_queue_size=10):
        """Starts the handler's workers.
        Args:
            workers: Number of workers.
            max_queue_size: queue size
                (when full, workers could block on `put()`)
        """
        if self.use_multiprocessing:
            self.executor_fn = self._get_executor_init(workers)
        else:
            # We do not need the init since it's threads.
            self.executor_fn = lambda _: get_pool_class(False)(workers)
        self.workers = workers
        self.queue = queue.Queue(max_queue_size)
        self.stop_signal = threading.Event()
        self.run_thread = threading.Thread(target=self._run)
        self.run_thread.daemon = True
        self.run_thread.start()
    def _send_sequence(self):
        """Sends current Iterable to all workers."""
        # For new processes that may spawn
        _SHARED_SEQUENCES[self.uid] = self.sequence
    def stop(self, timeout=None):
        """Stops running threads and wait for them to exit, if necessary.
        Should be called by the same thread which called `start()`.
        Args:
            timeout: maximum time to wait on `thread.join()`
        """
        self.stop_signal.set()
        with self.queue.mutex:
            self.queue.queue.clear()
            self.queue.unfinished_tasks = 0
            self.queue.not_full.notify()
        self.run_thread.join(timeout)
        _SHARED_SEQUENCES[self.uid] = None
    def __del__(self):
        if self.is_running():
            self.stop()
    @abstractmethod
    def _run(self):
        """Submits request to the executor and queue the `Future` objects."""
        raise NotImplementedError
    @abstractmethod
    def _get_executor_init(self, workers):
        """Gets the Pool initializer for multiprocessing.
        Args:
            workers: Number of workers.
        Returns:
            Function, a Function to initialize the pool
        """
        raise NotImplementedError
    @abstractmethod
    def get(self):
        """Creates a generator to extract data from the queue.
        Skip the data if it is `None`.
        # Returns
            Generator yielding tuples `(inputs, targets)`
                or `(inputs, targets, sample_weights)`.
        """
        raise NotImplementedError
