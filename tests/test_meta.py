"""Test suite for Meta Llama Collective Sentinel solution."""

import unittest
from meta_llama_collective_sentinel import MetaLlamaCollectiveSentinel


class TestMetaLlamaCollectiveSentinel(unittest.TestCase):
    def test_all_gather_optimization(self):
        sentinel = MetaLlamaCollectiveSentinel(
            gpu_count=1048576, fabric_bandwidth_tbps=400.0
        )
        res = sentinel.optimize_all_gather(
            parameter_size_bytes=70_000_000_000, tensor_parallel_size=8
        )

        self.assertEqual(res["status"], "ALL_GATHER_OPTIMAL")
        self.assertEqual(res["overhead_hidden_percent"], 100.0)


if __name__ == "__main__":
    unittest.main()
