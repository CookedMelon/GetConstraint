@tf_export("train.latest_checkpoint")
def latest_checkpoint(checkpoint_dir, latest_filename=None):
  """Finds the filename of latest saved checkpoint file.
  Gets the checkpoint state given the provided checkpoint_dir and looks for a
  corresponding TensorFlow 2 (preferred) or TensorFlow 1.x checkpoint path.
  The latest_filename argument is only applicable if you are saving checkpoint
  using `v1.train.Saver.save`
  See the [Training Checkpoints
  Guide](https://www.tensorflow.org/guide/checkpoint) for more details and
  examples.`
  Args:
    checkpoint_dir: Directory where the variables were saved.
    latest_filename: Optional name for the protocol buffer file that
      contains the list of most recent checkpoint filenames.
      See the corresponding argument to `v1.train.Saver.save`.
  Returns:
    The full path to the latest checkpoint or `None` if no checkpoint was found.
  """
  # Pick the latest checkpoint based on checkpoint state.
  ckpt = get_checkpoint_state(checkpoint_dir, latest_filename)
  if ckpt and ckpt.model_checkpoint_path:
    # Look for either a V2 path or a V1 path, with priority for V2.
    v2_path = _prefix_to_checkpoint_path(ckpt.model_checkpoint_path,
                                         saver_pb2.SaverDef.V2)
    v1_path = _prefix_to_checkpoint_path(ckpt.model_checkpoint_path,
                                         saver_pb2.SaverDef.V1)
    if file_io.get_matching_files(v2_path) or file_io.get_matching_files(
        v1_path):
      return ckpt.model_checkpoint_path
    else:
      logging.error("Couldn't match files for checkpoint %s",
                    ckpt.model_checkpoint_path)
  return None
