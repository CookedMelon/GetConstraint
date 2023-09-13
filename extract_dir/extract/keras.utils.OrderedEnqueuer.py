@keras_export("keras.utils.OrderedEnqueuer")
class OrderedEnqueuer(SequenceEnqueuer):
    """Builds a Enqueuer from a Sequence.
    Args:
        sequence: A `tf.keras.utils.data_utils.Sequence` object.
        use_multiprocessing: use multiprocessing if True, otherwise threading
        shuffle: whether to shuffle the data at the beginning of each epoch
    """
    def __init__(self, sequence, use_multiprocessing=False, shuffle=False):
        super().__init__(sequence, use_multiprocessing)
        self.shuffle = shuffle
    def _get_executor_init(self, workers):
        """Gets the Pool initializer for multiprocessing.
        Args:
            workers: Number of workers.
        Returns:
            Function, a Function to initialize the pool
        """
        def pool_fn(seqs):
            pool = get_pool_class(True)(
                workers,
                initializer=init_pool_generator,
                initargs=(seqs, None, get_worker_id_queue()),
            )
            _DATA_POOLS.add(pool)
            return pool
        return pool_fn
    def _wait_queue(self):
        """Wait for the queue to be empty."""
        while True:
            time.sleep(0.1)
            if self.queue.unfinished_tasks == 0 or self.stop_signal.is_set():
                return
    def _run(self):
        """Submits request to the executor and queue the `Future` objects."""
        sequence = list(range(len(self.sequence)))
        self._send_sequence()  # Share the initial sequence
        while True:
            if self.shuffle:
                random.shuffle(sequence)
            with closing(self.executor_fn(_SHARED_SEQUENCES)) as executor:
                for i in sequence:
                    if self.stop_signal.is_set():
                        return
                    self.queue.put(
                        executor.apply_async(get_index, (self.uid, i)),
                        block=True,
                    )
                # Done with the current epoch, waiting for the final batches
                self._wait_queue()
                if self.stop_signal.is_set():
                    # We're done
                    return
            # Call the internal on epoch end.
            self.sequence.on_epoch_end()
            self._send_sequence()  # Update the pool
    def get(self):
        """Creates a generator to extract data from the queue.
        Skip the data if it is `None`.
        Yields:
            The next element in the queue, i.e. a tuple
            `(inputs, targets)` or
            `(inputs, targets, sample_weights)`.
        """
        while self.is_running():
            try:
                inputs = self.queue.get(block=True, timeout=5).get()
                if self.is_running():
                    self.queue.task_done()
                if inputs is not None:
                    yield inputs
            except queue.Empty:
                pass
            except Exception as e:
                self.stop()
                raise e
