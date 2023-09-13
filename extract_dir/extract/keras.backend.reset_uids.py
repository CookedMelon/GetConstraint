@keras_export("keras.backend.reset_uids")
def reset_uids():
    """Resets graph identifiers."""
    PER_GRAPH_OBJECT_NAME_UIDS.clear()
    OBSERVED_NAMES.clear()
