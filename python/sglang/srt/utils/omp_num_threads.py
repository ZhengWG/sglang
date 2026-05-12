import math
import os
from typing import Optional


def _read_int_from_file(path: str) -> Optional[int]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    except (FileNotFoundError, ValueError, OSError):
        return None


def _quota_to_cpu_count(quota: int, period: int) -> Optional[int]:
    if quota <= 0 or period <= 0:
        return None
    return max(1, math.ceil(quota / period))


def _get_cgroup_v2_cpu_limit() -> Optional[int]:
    cpu_max_paths = ["/sys/fs/cgroup/cpu.max"]
    try:
        with open("/proc/self/cgroup", "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(":", maxsplit=2)
                if len(parts) == 3 and parts[0] == "0":
                    rel_path = parts[2].lstrip("/")
                    if rel_path:
                        cpu_max_paths.append(
                            os.path.join("/sys/fs/cgroup", rel_path, "cpu.max")
                        )
    except (FileNotFoundError, OSError):
        pass

    for cpu_max_path in cpu_max_paths:
        try:
            with open(cpu_max_path, "r", encoding="utf-8") as f:
                content = f.read().strip().split()
        except (FileNotFoundError, OSError):
            continue
        if len(content) != 2:
            continue
        quota_str, period_str = content
        if quota_str == "max":
            continue
        try:
            quota = int(quota_str)
            period = int(period_str)
        except ValueError:
            continue
        cpu_limit = _quota_to_cpu_count(quota, period)
        if cpu_limit is not None:
            return cpu_limit
    return None


def _get_cgroup_v1_cpu_limit() -> Optional[int]:
    base_paths = ["/sys/fs/cgroup/cpu", "/sys/fs/cgroup/cpu,cpuacct"]
    for base_path in base_paths:
        quota = _read_int_from_file(os.path.join(base_path, "cpu.cfs_quota_us"))
        period = _read_int_from_file(os.path.join(base_path, "cpu.cfs_period_us"))
        if quota is None or period is None:
            continue
        cpu_limit = _quota_to_cpu_count(quota, period)
        if cpu_limit is not None:
            return cpu_limit
    return None


def get_available_cpu_count() -> int:
    affinity_cpu_count: Optional[int] = None
    sched_getaffinity = getattr(os, "sched_getaffinity", None)
    if callable(sched_getaffinity):
        try:
            affinity_cpu_count = len(sched_getaffinity(0))
        except OSError:
            affinity_cpu_count = None

    cgroup_limit = _get_cgroup_v2_cpu_limit() or _get_cgroup_v1_cpu_limit()

    candidates = [
        v for v in (affinity_cpu_count, cgroup_limit) if v is not None and v > 0
    ]
    if candidates:
        return min(candidates)

    cpu_count = os.cpu_count()
    if cpu_count is None or cpu_count <= 0:
        return 1
    return cpu_count


def compute_default_omp_num_threads(
    available_cpu_count: int,
    local_process_count: int,
    reserve_cpu_count: int = 1,
) -> int:
    safe_cpu_count = max(1, available_cpu_count)
    safe_process_count = max(1, local_process_count)
    effective_cpu_count = max(1, safe_cpu_count - max(0, reserve_cpu_count))
    return max(1, effective_cpu_count // safe_process_count)
