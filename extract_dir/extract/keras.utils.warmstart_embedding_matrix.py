@keras_export("keras.utils.warmstart_embedding_matrix")
def warmstart_embedding_matrix(
    base_vocabulary,
    new_vocabulary,
    base_embeddings,
    new_embeddings_initializer="uniform",
