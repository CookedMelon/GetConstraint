@tf_export("nn.safe_embedding_lookup_sparse", v1=[])
@dispatch.add_dispatch_support
def safe_embedding_lookup_sparse_v2(
    embedding_weights,
    sparse_ids,
    sparse_weights=None,
    combiner="mean",
    default_id=None,
    max_norm=None,
    name=None,
    allow_fast_lookup=False,
