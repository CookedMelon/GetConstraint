@tf_export("saved_model.load", v1=["saved_model.load_v2"])
def load(export_dir, tags=None, options=None):
  """Load a SavedModel from `export_dir`.
  Signatures associated with the SavedModel are available as functions:
  ```python
  imported = tf.saved_model.load(path)
  f = imported.signatures["serving_default"]
  print(f(x=tf.constant([[1.]])))
  ```
  Objects exported with `tf.saved_model.save` additionally have trackable
  objects and functions assigned to attributes:
  ```python
  exported = tf.train.Checkpoint(v=tf.Variable(3.))
  exported.f = tf.function(
      lambda x: exported.v * x,
      input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
  tf.saved_model.save(exported, path)
  imported = tf.saved_model.load(path)
  assert 3. == imported.v.numpy()
  assert 6. == imported.f(x=tf.constant(2.)).numpy()
  ```
  _Loading Keras models_
  Keras models are trackable, so they can be saved to SavedModel. The object
  returned by `tf.saved_model.load` is not a Keras object (i.e. doesn't have
  `.fit`, `.predict`, etc. methods). A few attributes and functions are still
  available: `.variables`, `.trainable_variables` and `.__call__`.
  ```python
  model = tf.keras.Model(...)
  tf.saved_model.save(model, path)
  imported = tf.saved_model.load(path)
  outputs = imported(inputs)
  ```
  Use `tf.keras.models.load_model` to restore the Keras model.
  _Importing SavedModels from TensorFlow 1.x_
  SavedModels from `tf.estimator.Estimator` or 1.x SavedModel APIs have a flat
  graph instead of `tf.function` objects. These SavedModels will be loaded with
  the following attributes:
  * `.signatures`: A dictionary mapping signature names to functions.
  * `.prune(feeds, fetches) `: A method which allows you to extract
    functions for new subgraphs. This is equivalent to importing the SavedModel
    and naming feeds and fetches in a Session from TensorFlow 1.x.
    ```python
    imported = tf.saved_model.load(path_to_v1_saved_model)
    pruned = imported.prune("x:0", "out:0")
    pruned(tf.ones([]))
    ```
    See `tf.compat.v1.wrap_function` for details.
  * `.variables`: A list of imported variables.
  * `.graph`: The whole imported graph.
  * `.restore(save_path)`: A function that restores variables from a checkpoint
    saved from `tf.compat.v1.Saver`.
  _Consuming SavedModels asynchronously_
  When consuming SavedModels asynchronously (the producer is a separate
  process), the SavedModel directory will appear before all files have been
  written, and `tf.saved_model.load` will fail if pointed at an incomplete
  SavedModel. Rather than checking for the directory, check for
  "saved_model_dir/saved_model.pb". This file is written atomically as the last
  `tf.saved_model.save` file operation.
  Args:
    export_dir: The SavedModel directory to load from.
    tags: A tag or sequence of tags identifying the MetaGraph to load. Optional
      if the SavedModel contains a single MetaGraph, as for those exported from
      `tf.saved_model.save`.
    options: `tf.saved_model.LoadOptions` object that specifies options for
      loading.
  Returns:
    A trackable object with a `signatures` attribute mapping from signature
    keys to functions. If the SavedModel was exported by `tf.saved_model.save`,
    it also points to trackable objects, functions, debug info which it has been
    saved.
  Raises:
    ValueError: If `tags` don't match a MetaGraph in the SavedModel.
  """
  if isinstance(export_dir, os.PathLike):
    export_dir = os.fspath(export_dir)
  result = load_partial(export_dir, None, tags, options)["root"]
  return result
