#include <iostream>
#include <vector>

struct NCCLCollectiveMetric {
    size_t rank_id;
    double all_reduce_latency_us;
    double bandwidth_gbps;
};

class MetaLlamaCollectiveMonitor {
public:
    double compute_avg_bandwidth(const std::vector<NCCLCollectiveMetric>& metrics) {
        if (metrics.empty()) return 0.0;
        double sum = 0.0;
        for (const auto& m : metrics) {
            sum += m.bandwidth_gbps;
        }
        return sum / metrics.size();
    }
};

int main() {
    MetaLlamaCollectiveMonitor monitor;
    std::vector<NCCLCollectiveMetric> metrics{{0, 12.4, 850.0}, {1, 11.8, 860.0}};
    std::cout << "Average NCCL AllReduce Bandwidth: " << monitor.compute_avg_bandwidth(metrics) << " GB/s" << std::endl;
    return 0;
}
