import unittest
from unittest.mock import patch

from sglang.srt.utils.omp_num_threads import (
    compute_default_omp_num_threads,
    get_available_cpu_count,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


class TestOMPNumThreads(CustomTestCase):
    def test_compute_default_omp_num_threads(self):
        self.assertEqual(
            compute_default_omp_num_threads(
                available_cpu_count=32,
                local_process_count=8,
            ),
            3,
        )
        self.assertEqual(
            compute_default_omp_num_threads(
                available_cpu_count=2,
                local_process_count=8,
            ),
            1,
        )

    @patch("sglang.srt.utils.omp_num_threads._get_cgroup_v2_cpu_limit", return_value=6)
    @patch("sglang.srt.utils.omp_num_threads._get_cgroup_v1_cpu_limit", return_value=None)
    @patch("sglang.srt.utils.omp_num_threads.os.sched_getaffinity")
    def test_get_available_cpu_count_prefers_cgroup_limit(
        self, mock_affinity, _mock_v1, _mock_v2
    ):
        mock_affinity.return_value = set(range(16))
        self.assertEqual(get_available_cpu_count(), 6)

    @patch("sglang.srt.utils.omp_num_threads._get_cgroup_v2_cpu_limit", return_value=None)
    @patch("sglang.srt.utils.omp_num_threads._get_cgroup_v1_cpu_limit", return_value=None)
    @patch("sglang.srt.utils.omp_num_threads.os.sched_getaffinity")
    def test_get_available_cpu_count_uses_affinity(
        self, mock_affinity, _mock_v1, _mock_v2
    ):
        mock_affinity.return_value = {0, 1, 2, 3}
        self.assertEqual(get_available_cpu_count(), 4)

    @patch("sglang.srt.utils.omp_num_threads._get_cgroup_v2_cpu_limit", return_value=None)
    @patch("sglang.srt.utils.omp_num_threads._get_cgroup_v1_cpu_limit", return_value=None)
    @patch("sglang.srt.utils.omp_num_threads.os.cpu_count", return_value=12)
    @patch("sglang.srt.utils.omp_num_threads.os.sched_getaffinity")
    def test_get_available_cpu_count_falls_back_to_cpu_count(
        self, mock_affinity, _mock_cpu_count, _mock_v1, _mock_v2
    ):
        mock_affinity.side_effect = OSError("affinity unavailable")
        self.assertEqual(get_available_cpu_count(), 12)


if __name__ == "__main__":
    unittest.main()
