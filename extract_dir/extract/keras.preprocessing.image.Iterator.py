@keras_export("keras.preprocessing.image.Iterator")
class Iterator(data_utils.Sequence):
    """Base class for image data iterators.
    Deprecated: `tf.keras.preprocessing.image.Iterator` is not recommended for
    new code. Prefer loading images with
    `tf.keras.utils.image_dataset_from_directory` and transforming the output
    `tf.data.Dataset` with preprocessing layers. For more information, see the
    tutorials for [loading images](
    https://www.tensorflow.org/tutorials/load_data/images) and
    [augmenting images](
    https://www.tensorflow.org/tutorials/images/data_augmentation), as well as
    the [preprocessing layer guide](
    https://www.tensorflow.org/guide/keras/preprocessing_layers).
    Every `Iterator` must implement the `_get_batches_of_transformed_samples`
    method.
    Args:
        n: Integer, total number of samples in the dataset to loop over.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seeding for data shuffling.
    """
    white_list_formats = ("png", "jpg", "jpeg", "bmp", "ppm", "tif", "tiff")
    def __init__(self, n, batch_size, shuffle, seed):
        self.n = n
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_array = None
        self.index_generator = self._flow_index()
    def _set_index_array(self):
        self.index_array = np.arange(self.n)
        if self.shuffle:
            self.index_array = np.random.permutation(self.n)
    def __getitem__(self, idx):
        if idx >= len(self):
            raise ValueError(
                "Asked to retrieve element {idx}, "
                "but the Sequence "
                "has length {length}".format(idx=idx, length=len(self))
            )
        if self.seed is not None:
            np.random.seed(self.seed + self.total_batches_seen)
        self.total_batches_seen += 1
        if self.index_array is None:
            self._set_index_array()
        index_array = self.index_array[
            self.batch_size * idx : self.batch_size * (idx + 1)
        ]
        return self._get_batches_of_transformed_samples(index_array)
    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size  # round up
    def on_epoch_end(self):
        self._set_index_array()
    def reset(self):
        self.batch_index = 0
    def _flow_index(self):
        # Ensure self.batch_index is 0.
        self.reset()
        while 1:
            if self.seed is not None:
                np.random.seed(self.seed + self.total_batches_seen)
            if self.batch_index == 0:
                self._set_index_array()
            if self.n == 0:
                # Avoiding modulo by zero error
                current_index = 0
            else:
                current_index = (self.batch_index * self.batch_size) % self.n
            if self.n > current_index + self.batch_size:
                self.batch_index += 1
            else:
                self.batch_index = 0
            self.total_batches_seen += 1
            yield self.index_array[
                current_index : current_index + self.batch_size
            ]
    def __iter__(self):
        # Needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self
    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)
    def next(self):
        """For python 2.x.
        Returns:
            The next batch.
        """
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)
    def _get_batches_of_transformed_samples(self, index_array):
        """Gets a batch of transformed samples.
        Args:
            index_array: Array of sample indices to include in batch.
        Returns:
            A batch of transformed samples.
        """
        raise NotImplementedError
