@tf_export("saved_model.experimental.read_fingerprint", v1=[])
def read_fingerprint(export_dir):
  """Reads the fingerprint of a SavedModel in `export_dir`.
  Returns a `tf.saved_model.experimental.Fingerprint` object that contains
  the values of the SavedModel fingerprint, which is persisted on disk in the
  `fingerprint.pb` file in the `export_dir`.
  Read more about fingerprints in the SavedModel guide at
  https://www.tensorflow.org/guide/saved_model.
  Args:
    export_dir: The directory that contains the SavedModel.
  Returns:
    A `tf.saved_model.experimental.Fingerprint`.
  Raises:
    FileNotFoundError: If no or an invalid fingerprint is found.
  """
  try:
    fingerprint = fingerprinting_pywrap.ReadSavedModelFingerprint(export_dir)
  except fingerprinting_pywrap.FileNotFoundException as e:
    raise FileNotFoundError(f"SavedModel Fingerprint Error: {e}") from None  # pylint: disable=raise-missing-from
  except fingerprinting_pywrap.FingerprintException as e:
    raise RuntimeError(f"SavedModel Fingerprint Error: {e}") from None  # pylint: disable=raise-missing-from
  return Fingerprint.from_proto(
      fingerprint_pb2.FingerprintDef().FromString(fingerprint))
