@tf_export("data.experimental.sample_from_datasets", v1=[])
def sample_from_datasets_v2(datasets,
                            weights=None,
                            seed=None,
                            stop_on_empty_dataset=False):
  """Samples elements at random from the datasets in `datasets`.
  Creates a dataset by interleaving elements of `datasets` with `weight[i]`
  probability of picking an element from dataset `i`. Sampling is done without
  replacement. For example, suppose we have 2 datasets:
  ```python
  dataset1 = tf.data.Dataset.range(0, 3)
  dataset2 = tf.data.Dataset.range(100, 103)
  ```
  Suppose also that we sample from these 2 datasets with the following weights:
  ```python
  sample_dataset = tf.data.Dataset.sample_from_datasets(
      [dataset1, dataset2], weights=[0.5, 0.5])
  ```
  One possible outcome of elements in sample_dataset is:
  ```
  print(list(sample_dataset.as_numpy_iterator()))
  # [100, 0, 1, 101, 2, 102]
  ```
  Args:
    datasets: A non-empty list of `tf.data.Dataset` objects with compatible
      structure.
    weights: (Optional.) A list or Tensor of `len(datasets)` floating-point
      values where `weights[i]` represents the probability to sample from
      `datasets[i]`, or a `tf.data.Dataset` object where each element is such a
      list. Defaults to a uniform distribution across `datasets`.
    seed: (Optional.) A `tf.int64` scalar `tf.Tensor`, representing the random
      seed that will be used to create the distribution. See
      `tf.random.set_seed` for behavior.
    stop_on_empty_dataset: If `True`, sampling stops if it encounters an empty
      dataset. If `False`, it skips empty datasets. It is recommended to set it
      to `True`. Otherwise, the distribution of samples starts off as the user
      intends, but may change as input datasets become empty. This can be
      difficult to detect since the dataset starts off looking correct. Default
      to `False` for backward compatibility.
  Returns:
    A dataset that interleaves elements from `datasets` at random, according to
    `weights` if provided, otherwise with uniform probability.
  Raises:
    TypeError: If the `datasets` or `weights` arguments have the wrong type.
    ValueError:
      - If `datasets` is empty, or
      - If `weights` is specified and does not match the length of `datasets`.
  """
  return dataset_ops.Dataset.sample_from_datasets(
      datasets=datasets,
      weights=weights,
      seed=seed,
      stop_on_empty_dataset=stop_on_empty_dataset)
