"""Test suite for Meta Llama Collective Monitor."""

import unittest


class MetaLlamaCollectiveMonitorSim:
    def compute_avg_bandwidth(self, bandwidths: list) -> float:
        return sum(bandwidths) / max(len(bandwidths), 1)


class TestMetaLlamaCollectiveMonitor(unittest.TestCase):
    def test_bandwidth_avg(self):
        sim = MetaLlamaCollectiveMonitorSim()
        avg = sim.compute_avg_bandwidth([850.0, 860.0])
        self.assertEqual(avg, 855.0)


if __name__ == "__main__":
    unittest.main()
