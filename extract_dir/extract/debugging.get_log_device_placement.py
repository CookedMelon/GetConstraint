@tf_export("debugging.get_log_device_placement")
def get_log_device_placement():
  """Get if device placements are logged.
  Returns:
    If device placements are logged.
  """
  return context().log_device_placement
