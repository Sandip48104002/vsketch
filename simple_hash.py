import numpy as np

# ==============================================================================
# Convert IPv4 string to 32-bit integer
# ==============================================================================
def ip_to_int(ip):
    a, b, c, d = map(int, ip.split('.'))
    return (a << 24) | (b << 16) | (c << 8) | d


# ==============================================================================
# Traffic generation (Zipf, 1 packet per flow guaranteed)
# ==============================================================================
def generate_ip_traffic(
    n_flows=50_000,
    n_packets=30_000_000,
    zipf_param=1.2,
    seed=42
):
    rng = np.random.default_rng(seed)

    ip_ints = rng.choice(2**32, size=n_flows, replace=False)
    ips = np.array(
        [f"{(x>>24)&255}.{(x>>16)&255}.{(x>>8)&255}.{x&255}" for x in ip_ints],
        dtype=object
    )

    weights = rng.zipf(zipf_param, n_flows).astype(np.float64)
    weights /= weights.sum()

    packets = np.empty(n_packets, dtype=object)
    packets[:n_flows] = ips
    packets[n_flows:] = rng.choice(
        ips, size=n_packets - n_flows, p=weights
    )
    rng.shuffle(packets)

    # True counts
    true_counts = {ip: 1 for ip in ips}
    for ip in packets[n_flows:]:
        true_counts[ip] += 1

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

    def insert(self, ip):
        col = ip_to_int(ip) % self.width
        for r in range(self.rows):
            if ip not in self.seen[r][col] and self.C[r][col] != 0:
                self.collisions += 1
            self.seen[r][col].add(ip)
            self.C[r][col] += 1

    def query(self, ip):
        col = ip_to_int(ip) % self.width
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
        self.s = 0

    def insert(self, ip):
        self.s += 1
        base = ip_to_int(ip)
        v = (base + self.s) % self.k
        col = (base + v) % self.width

        for r in range(self.rows):
            if ip not in self.seen[r][col] and self.C[r][col] != 0:
                self.collisions += 1
            self.seen[r][col].add(ip)
            self.C[r][col] += 1

    def query(self, ip):
        base = ip_to_int(ip)
        total = 0
        for v in range(self.k):
            col = (base + v) % self.width
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

    packets, true_counts = generate_ip_traffic(n_flows, n_packets)

    cms = CMS_Simple(rows, width)
    var = CMS_Variant_Simple(rows, width, k)

    for ip in packets:
        cms.insert(ip)
        var.insert(ip)

    cms_errors = [cms.query(ip) - true_counts[ip] for ip in true_counts]
    var_errors = [var.query(ip) - true_counts[ip] for ip in true_counts]

    print("\n=== Simple MOD Hash vs Variant MOD Hash ===")
    print(f"Flows      : {n_flows}")
    print(f"Packets    : {n_packets}")
    print(f"Rows       : {rows}")
    print(f"Width      : {width} (2^14)")
    print(f"k          : {k}")
    print("-----------------------------------------")
    print(f"Simple CMS  | Mean Error: {np.mean(cms_errors):.2f} | "
          f"Util: {cms.utilization()*100:.2f}% | Collisions: {cms.collisions}")
    print(f"Variant CMS | Mean Error: {np.mean(var_errors):.2f} | "
          f"Util: {var.utilization()*100:.2f}% | Collisions: {var.collisions}")


# ==============================================================================
# Main
# ==============================================================================
if __name__ == "__main__":
    run_experiment()
