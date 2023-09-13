@tf_export("distribute.ReduceOp")
class ReduceOp(enum.Enum):
  """Indicates how a set of values should be reduced.
  * `SUM`: Add all the values.
  * `MEAN`: Take the arithmetic mean ("average") of the values.
  """
  # TODO(priyag): Add the following types:
  # `MIN`: Return the minimum of all values.
  # `MAX`: Return the maximum of all values.
  SUM = "SUM"
  MEAN = "MEAN"
  @staticmethod
  def from_variable_aggregation(aggregation):
    mapping = {
        variable_scope.VariableAggregation.SUM: ReduceOp.SUM,
        variable_scope.VariableAggregation.MEAN: ReduceOp.MEAN,
    }
    reduce_op = mapping.get(aggregation)
    if not reduce_op:
      raise ValueError("Could not convert from `tf.VariableAggregation` %s to"
                       "`tf.distribute.ReduceOp` type" % aggregation)
    return reduce_op
