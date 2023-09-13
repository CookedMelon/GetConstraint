@tf_export("train.get_checkpoint_state")
def get_checkpoint_state(checkpoint_dir, latest_filename=None):
  """Returns CheckpointState proto from the "checkpoint" file.
  If the "checkpoint" file contains a valid CheckpointState
  proto, returns it.
  Args:
    checkpoint_dir: The directory of checkpoints.
    latest_filename: Optional name of the checkpoint file.  Default to
      'checkpoint'.
  Returns:
    A CheckpointState if the state was available, None
    otherwise.
  Raises:
    ValueError: if the checkpoint read doesn't have model_checkpoint_path set.
  """
  if isinstance(checkpoint_dir, os.PathLike):
    checkpoint_dir = os.fspath(checkpoint_dir)
  ckpt = None
  coord_checkpoint_filename = _GetCheckpointFilename(checkpoint_dir,
                                                     latest_filename)
  f = None
  try:
    # Check that the file exists before opening it to avoid
    # many lines of errors from colossus in the logs.
    if file_io.file_exists(coord_checkpoint_filename):
      file_content = file_io.read_file_to_string(
          coord_checkpoint_filename)
      ckpt = CheckpointState()
      text_format.Merge(file_content, ckpt)
      if not ckpt.model_checkpoint_path:
        raise ValueError("Invalid checkpoint state loaded from "
                         + checkpoint_dir)
      # For relative model_checkpoint_path and all_model_checkpoint_paths,
      # prepend checkpoint_dir.
      if not os.path.isabs(ckpt.model_checkpoint_path):
        ckpt.model_checkpoint_path = os.path.join(checkpoint_dir,
                                                  ckpt.model_checkpoint_path)
      for i, p in enumerate(ckpt.all_model_checkpoint_paths):
        if not os.path.isabs(p):
          ckpt.all_model_checkpoint_paths[i] = os.path.join(checkpoint_dir, p)
  except errors.OpError as e:
    # It's ok if the file cannot be read
    logging.warning("%s: %s", type(e).__name__, e)
    logging.warning("%s: Checkpoint ignored", coord_checkpoint_filename)
    return None
  except text_format.ParseError as e:
    logging.warning("%s: %s", type(e).__name__, e)
    logging.warning("%s: Checkpoint ignored", coord_checkpoint_filename)
    return None
  finally:
    if f:
      f.close()
  return ckpt
