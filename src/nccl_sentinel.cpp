/**
 * Meta Llama Collective Sentinel — C++ NCCL All-Reduce Straggler Detector
 * Monitors GPU-to-GPU NCCL communication bandwidth and identifies slow straggler ranks.
 */

#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>

struct NCCLRankMetric {
    size_t rank_id;
    double latency_us;
    double bandwidth_gbps;
};

class NCCLCollectiveMonitor {
public:
    double compute_avg_bandwidth(const std::vector<NCCLRankMetric>& metrics) {
        if (metrics.empty()) return 0.0;
        double sum = 0.0;
        for (const auto& m : metrics) sum += m.bandwidth_gbps;
        return sum / metrics.size();
    }

    std::vector<size_t> detect_stragglers(const std::vector<NCCLRankMetric>& metrics, double threshold_ratio = 0.7) {
        double avg = compute_avg_bandwidth(metrics);
        std::vector<size_t> stragglers;
        for (const auto& m : metrics) {
            if (m.bandwidth_gbps < avg * threshold_ratio) {
                stragglers.push_back(m.rank_id);
            }
        }
        return stragglers;
    }
};

int main() {
    NCCLCollectiveMonitor monitor;
    std::vector<NCCLRankMetric> metrics = {
        {0, 12.5, 850.0},
        {1, 12.8, 840.0},
        {2, 45.0, 210.0}, // Straggler
        {3, 12.6, 845.0}
    };

    auto stragglers = monitor.detect_stragglers(metrics);
    std::cout << "[NCCLSentinel] Avg Bandwidth: " << monitor.compute_avg_bandwidth(metrics) << " GB/s" << std::endl;
    std::cout << "[NCCLSentinel] Detected " << stragglers.size() << " straggler rank(s)." << std::endl;
    return 0;
}
