@tf_export("config.experimental_functions_run_eagerly")
def experimental_functions_run_eagerly():
  """Returns the value of the `experimental_run_functions_eagerly` setting."""
  return eager_function_run.functions_run_eagerly()
