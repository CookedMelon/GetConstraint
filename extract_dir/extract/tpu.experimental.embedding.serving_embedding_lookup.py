@tf_export("tpu.experimental.embedding.serving_embedding_lookup")
def cpu_embedding_lookup(
    inputs: Any,
    weights: Optional[Any],
    tables: Dict[tpu_embedding_v2_utils.TableConfig, tf_variables.Variable],
    feature_config: Union[tpu_embedding_v2_utils.FeatureConfig, Iterable]  # pylint:disable=g-bare-generic
