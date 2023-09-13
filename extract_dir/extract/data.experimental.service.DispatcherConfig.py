@tf_export("data.experimental.service.DispatcherConfig")
class DispatcherConfig(
    collections.namedtuple(
        "DispatcherConfig",
        [
            "port",
            "protocol",
            "work_dir",
            "fault_tolerant_mode",
            "worker_addresses",
            "job_gc_check_interval_ms",
            "job_gc_timeout_ms",
            "worker_timeout_ms",
        ],
    )
