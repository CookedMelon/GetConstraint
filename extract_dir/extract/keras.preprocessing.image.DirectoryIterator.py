@keras_export("keras.preprocessing.image.DirectoryIterator")
class DirectoryIterator(BatchFromFilesMixin, Iterator):
    """Iterator capable of reading images from a directory on disk.
    Deprecated: `tf.keras.preprocessing.image.DirectoryIterator` is not
    recommended for new code. Prefer loading images with
    `tf.keras.utils.image_dataset_from_directory` and transforming the output
    `tf.data.Dataset` with preprocessing layers. For more information, see the
    tutorials for [loading images](
    https://www.tensorflow.org/tutorials/load_data/images) and
    [augmenting images](
    https://www.tensorflow.org/tutorials/images/data_augmentation), as well as
    the [preprocessing layer guide](
    https://www.tensorflow.org/guide/keras/preprocessing_layers).
    Args:
        directory: Path to the directory to read images from. Each subdirectory
          in this directory will be considered to contain images from one class,
          or alternatively you could specify class subdirectories via the
          `classes` argument.
        image_data_generator: Instance of `ImageDataGenerator` to use for random
          transformations and normalization.
        target_size: tuple of integers, dimensions to resize input images to.
        color_mode: One of `"rgb"`, `"rgba"`, `"grayscale"`. Color mode to read
          images.
        classes: Optional list of strings, names of subdirectories containing
          images from each class (e.g. `["dogs", "cats"]`). It will be computed
          automatically if not set.
        class_mode: Mode for yielding the targets:
            - `"binary"`: binary targets (if there are only two classes),
            - `"categorical"`: categorical targets,
            - `"sparse"`: integer targets,
            - `"input"`: targets are images identical to input images (mainly
              used to work with autoencoders),
            - `None`: no targets get yielded (only input images are yielded).
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        save_to_dir: Optional directory where to save the pictures being
          yielded, in a viewable format. This is useful for visualizing the
          random transformations being applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample images (if
          `save_to_dir` is set).
        save_format: Format to use for saving sample images (if `save_to_dir` is
          set).
        subset: Subset of data (`"training"` or `"validation"`) if
          validation_split is set in ImageDataGenerator.
        interpolation: Interpolation method used to resample the image if the
          target size is different from that of the loaded image. Supported
          methods are "nearest", "bilinear", and "bicubic". If PIL version 1.1.3
          or newer is installed, "lanczos" is also supported. If PIL version
          3.4.0 or newer is installed, "box" and "hamming" are also supported.
          By default, "nearest" is used.
        keep_aspect_ratio: Boolean, whether to resize images to a target size
            without aspect ratio distortion. The image is cropped in the center
            with target aspect ratio before resizing.
        dtype: Dtype to use for generated arrays.
    """
    allowed_class_modes = {"categorical", "binary", "sparse", "input", None}
    def __init__(
        self,
        directory,
        image_data_generator,
        target_size=(256, 256),
        color_mode="rgb",
        classes=None,
        class_mode="categorical",
        batch_size=32,
        shuffle=True,
        seed=None,
        data_format=None,
        save_to_dir=None,
        save_prefix="",
        save_format="png",
        follow_links=False,
        subset=None,
        interpolation="nearest",
        keep_aspect_ratio=False,
        dtype=None,
    ):
        if data_format is None:
            data_format = backend.image_data_format()
        if dtype is None:
            dtype = backend.floatx()
        super().set_processing_attrs(
            image_data_generator,
            target_size,
            color_mode,
            data_format,
            save_to_dir,
            save_prefix,
            save_format,
            subset,
            interpolation,
            keep_aspect_ratio,
        )
        self.directory = directory
        self.classes = classes
        if class_mode not in self.allowed_class_modes:
            raise ValueError(
                "Invalid class_mode: {}; expected one of: {}".format(
                    class_mode, self.allowed_class_modes
                )
            )
        self.class_mode = class_mode
        self.dtype = dtype
        # First, count the number of samples and classes.
        self.samples = 0
        if not classes:
            classes = []
            for subdir in sorted(os.listdir(directory)):
                if os.path.isdir(os.path.join(directory, subdir)):
                    classes.append(subdir)
        self.num_classes = len(classes)
        self.class_indices = dict(zip(classes, range(len(classes))))
        pool = multiprocessing.pool.ThreadPool()
        # Second, build an index of the images
        # in the different class subfolders.
        results = []
        self.filenames = []
        i = 0
        for dirpath in (os.path.join(directory, subdir) for subdir in classes):
            results.append(
                pool.apply_async(
                    _list_valid_filenames_in_directory,
                    (
                        dirpath,
                        self.white_list_formats,
                        self.split,
                        self.class_indices,
                        follow_links,
                    ),
                )
            )
        classes_list = []
        for res in results:
            classes, filenames = res.get()
            classes_list.append(classes)
            self.filenames += filenames
        self.samples = len(self.filenames)
        self.classes = np.zeros((self.samples,), dtype="int32")
        for classes in classes_list:
            self.classes[i : i + len(classes)] = classes
            i += len(classes)
        io_utils.print_msg(
            f"Found {self.samples} images belonging to "
            f"{self.num_classes} classes."
        )
        pool.close()
        pool.join()
        self._filepaths = [
            os.path.join(self.directory, fname) for fname in self.filenames
        ]
        super().__init__(self.samples, batch_size, shuffle, seed)
    @property
    def filepaths(self):
        return self._filepaths
    @property
    def labels(self):
        return self.classes
    @property  # mixin needs this property to work
    def sample_weight(self):
        # no sample weights will be returned
        return None
