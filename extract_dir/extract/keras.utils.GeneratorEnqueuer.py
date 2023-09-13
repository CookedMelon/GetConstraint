@keras_export("keras.utils.GeneratorEnqueuer")
class GeneratorEnqueuer(SequenceEnqueuer):
    """Builds a queue out of a data generator.
    The provided generator can be finite in which case the class will throw
    a `StopIteration` exception.
    Args:
        generator: a generator function which yields data
        use_multiprocessing: use multiprocessing if True, otherwise threading
        random_seed: Initial seed for workers,
            will be incremented by one for each worker.
    """
    def __init__(self, generator, use_multiprocessing=False, random_seed=None):
        super().__init__(generator, use_multiprocessing)
        self.random_seed = random_seed
    def _get_executor_init(self, workers):
        """Gets the Pool initializer for multiprocessing.
        Args:
          workers: Number of works.
        Returns:
            A Function to initialize the pool
        """
        def pool_fn(seqs):
            pool = get_pool_class(True)(
                workers,
                initializer=init_pool_generator,
                initargs=(seqs, self.random_seed, get_worker_id_queue()),
            )
            _DATA_POOLS.add(pool)
            return pool
        return pool_fn
    def _run(self):
        """Submits request to the executor and queue the `Future` objects."""
        self._send_sequence()  # Share the initial generator
        with closing(self.executor_fn(_SHARED_SEQUENCES)) as executor:
            while True:
                if self.stop_signal.is_set():
                    return
                self.queue.put(
                    executor.apply_async(next_sample, (self.uid,)), block=True
                )
    def get(self):
        """Creates a generator to extract data from the queue.
        Skip the data if it is `None`.
        Yields:
            The next element in the queue, i.e. a tuple
            `(inputs, targets)` or
            `(inputs, targets, sample_weights)`.
        """
        try:
            while self.is_running():
                inputs = self.queue.get(block=True).get()
                self.queue.task_done()
                if inputs is not None:
                    yield inputs
        except StopIteration:
            # Special case for finite generators
            last_ones = []
            while self.queue.qsize() > 0:
                last_ones.append(self.queue.get(block=True))
            # Wait for them to complete
            for f in last_ones:
                f.wait()
            # Keep the good ones
            last_ones = [
                future.get() for future in last_ones if future.successful()
            ]
            for inputs in last_ones:
                if inputs is not None:
                    yield inputs
        except Exception as e:
            self.stop()
            if "generator already executing" in str(e):
                raise RuntimeError(
                    "Your generator is NOT thread-safe. "
                    "Keras requires a thread-safe generator when "
                    "`use_multiprocessing=False, workers > 1`. "
                )
            raise e
