import numpy as np

# ==============================================================================
# Traffic generation: Sequential flow IDs
# ==============================================================================
def generate_flow_traffic(
    n_flows=50_000,
    n_packets=30_000_000,
    zipf_param=1.2,
    seed=42
):
    rng = np.random.default_rng(seed)

    # Flow IDs: 1, 2, 3, ..., 50000
    flows = np.arange(1, n_flows + 1, dtype=np.int64)

    # Zipf weights
    weights = rng.zipf(zipf_param, n_flows).astype(np.float64)
    weights /= weights.sum()

    # Packet stream
    packets = np.empty(n_packets, dtype=np.int64)
    packets[:n_flows] = flows                          # ensure 1 packet per flow
    packets[n_flows:] = rng.choice(
        flows, size=n_packets - n_flows, p=weights
    )
    rng.shuffle(packets)

    # True counts
    true_counts = {int(f): 1 for f in flows}
    for f in packets[n_flows:]:
        true_counts[int(f)] += 1

    return packets, true_counts


# ==============================================================================
# SIMPLE CMS (MOD hash)
# ==============================================================================
class CMS_Simple:
    def __init__(self, rows=4, width=2**14):
        self.rows = rows
        self.width = width
        self.C = [[0]*width for _ in range(rows)]
        self.seen = [[set() for _ in range(width)] for _ in range(rows)]
        self.collisions = 0

    def insert(self, flow_id):
        col = flow_id % self.width
        for r in range(self.rows):
            if flow_id not in self.seen[r][col] and self.C[r][col] != 0:
                self.collisions += 1
            self.seen[r][col].add(flow_id)
            self.C[r][col] += 1

    def query(self, flow_id):
        col = flow_id % self.width
        return min(self.C[r][col] for r in range(self.rows))

    def utilization(self):
        C_np = np.array(self.C)
        return np.count_nonzero(C_np) / C_np.size


# ==============================================================================
# VARIANT CMS (MOD hash)
# ==============================================================================
class CMS_Variant_Simple:
    def __init__(self, rows=4, width=2**14, k=3):
        self.rows = rows
        self.width = width
        self.k = k
        self.C = [[0]*width for _ in range(rows)]
        self.seen = [[set() for _ in range(width)] for _ in range(rows)]
        self.collisions = 0
        self.s = 0  # global packet counter

    def insert(self, flow_id):
        self.s += 1
        v = (flow_id + self.s) % self.k
        col = (flow_id + v) % self.width

        for r in range(self.rows):
            if flow_id not in self.seen[r][col] and self.C[r][col] != 0:
                self.collisions += 1
            self.seen[r][col].add(flow_id)
            self.C[r][col] += 1

    def query(self, flow_id):
        total = 0
        for v in range(self.k):
            col = (flow_id + v) % self.width
            total += min(self.C[r][col] for r in range(self.rows))
        return total

    def utilization(self):
        C_np = np.array(self.C)
        return np.count_nonzero(C_np) / C_np.size


# ==============================================================================
# Experiment
# ==============================================================================
def run_experiment():
    rows = 4
    width = 2**14
    k = 3
    n_flows = 50_000
    n_packets = 30_000_000

    packets, true_counts = generate_flow_traffic(n_flows, n_packets)

    cms = CMS_Simple(rows, width)
    var = CMS_Variant_Simple(rows, width, k)

    for f in packets:
        cms.insert(f)
        var.insert(f)

    cms_errors = [cms.query(f) - true_counts[f] for f in true_counts]
    var_errors = [var.query(f) - true_counts[f] for f in true_counts]

    print("\n=== Sequential Flows: Simple vs Variant CMS (MOD hash) ===")
    print(f"Flows      : {n_flows} (1..{n_flows})")
    print(f"Packets    : {n_packets}")
    print(f"Rows       : {rows}")
    print(f"Width      : {width} (2^14)")
    print(f"k          : {k}")
    print("---------------------------------------------------------")
    print(f"Simple CMS  | Mean Error: {np.mean(cms_errors):.2f} | "
          f"Util: {cms.utilization()*100:.2f}% | Collisions: {cms.collisions}")
    print(f"Variant CMS | Mean Error: {np.mean(var_errors):.2f} | "
          f"Util: {var.utilization()*100:.2f}% | Collisions: {var.collisions}")


# ==============================================================================
# Main
# ==============================================================================
if __name__ == "__main__":
    run_experiment()
