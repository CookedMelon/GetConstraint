@tf_export("train.load_variable")
def load_variable(ckpt_dir_or_file, name):
  """Returns the tensor value of the given variable in the checkpoint.
  When the variable name is unknown, you can use `tf.train.list_variables` to
  inspect all the variable names.
  Example usage:
  ```python
  import tensorflow as tf
  a = tf.Variable(1.0)
  b = tf.Variable(2.0)
  ckpt = tf.train.Checkpoint(var_list={'a': a, 'b': b})
  ckpt_path = ckpt.save('tmp-ckpt')
  var= tf.train.load_variable(
      ckpt_path, 'var_list/a/.ATTRIBUTES/VARIABLE_VALUE')
  print(var)  # 1.0
  ```
  Args:
    ckpt_dir_or_file: Directory with checkpoints file or path to checkpoint.
    name: Name of the variable to return.
  Returns:
    A numpy `ndarray` with a copy of the value of this variable.
  """
  # TODO(b/29227106): Fix this in the right place and remove this.
  if name.endswith(":0"):
    name = name[:-2]
  reader = load_checkpoint(ckpt_dir_or_file)
  return reader.get_tensor(name)
