@tf_export("nn.embedding_lookup_sparse", v1=[])
@dispatch.add_dispatch_support
def embedding_lookup_sparse_v2(
    params,
    sp_ids,
    sp_weights,
    combiner=None,
    max_norm=None,
    name=None,
    allow_fast_lookup=False,
