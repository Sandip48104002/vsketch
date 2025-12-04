import random
import xxhash
import numpy as np
import matplotlib.pyplot as plt


# ================================================================
# Standard Count-Min Sketch
# ================================================================
class CountMinSketch:
    def __init__(self, d, w):
        self.d = d
        self.w = w
        self.table = np.zeros((d, w), dtype=np.int64)
        self.hashes = [xxhash.xxh64(seed=i) for i in range(d)]

    def insert(self, key):
        for i in range(self.d):
            h = xxhash.xxh64(key, seed=i).intdigest() % self.w
            self.table[i, h] += 1

    def query(self, key):
        vals = []
        for i in range(self.d):
            h = xxhash.xxh64(key, seed=i).intdigest() % self.w
            vals.append(self.table[i, h])
        return min(vals)


# ================================================================
# Variant-Key CMS
# ================================================================
class VariantCMS:
    def __init__(self, d, w, k):
        self.d = d
        self.w = w
        self.k = k
        self.table = np.zeros((d, w), dtype=np.int64)
        self.packet_counter = 0

    def variant_key(self, ip):
        self.packet_counter += 1
        v = (xxhash.xxh64(ip).intdigest() + self.packet_counter) % self.k
        return f"{ip}:{v}"

    def insert(self, ip):
        key = self.variant_key(ip)
        for i in range(self.d):
            h = xxhash.xxh64(key, seed=i).intdigest() % self.w
            self.table[i, h] += 1

    def query(self, ip):
        total = 0
        for v in range(self.k):
            key = f"{ip}:{v}"
            vals = []
            for i in range(self.d):
                h = xxhash.xxh64(key, seed=i).intdigest() % self.w
                vals.append(self.table[i, h])
            total += min(vals)
        return total


# ================================================================
# Traffic Generator
# ================================================================
def generate_flows(n_flows, n_packets):
    flows = [f"IP{i}" for i in range(n_flows)]
    weights = np.random.zipf(1.2, n_flows)  # Realistic skew
    weights = weights / weights.sum()

    packets = np.random.choice(flows, size=n_packets, p=weights)
    true_counts = {f: 0 for f in flows}
    for p in packets:
        true_counts[p] += 1
    return packets, true_counts


# ================================================================
# Evaluation
# ================================================================
def evaluate(d=3, w=20000, k=3, n_flows=10000, n_packets=300000):
    packets, TRUE = generate_flows(n_flows, n_packets)

    cms = CountMinSketch(d, w)
    vcms = VariantCMS(d, w, k)

    for p in packets:
        cms.insert(p)
        vcms.insert(p)

    errors_std = []
    errors_var = []

    for ip, t in TRUE.items():
        s = cms.query(ip)
        v = vcms.query(ip)
        errors_std.append(s - t)
        errors_var.append(v - t)

    return np.array(errors_std), np.array(errors_var), TRUE


# ================================================================
# Run Experiments for Different Sketch Sizes
# ================================================================
def sweep_widths(widths, n_flows=5000, n_packets=200000):
    print(f"\nRunning width sweep: widths={widths}")

    results = []

    for w in widths:
        print(f"Testing width w={w}...")
        err_std, err_var, _ = evaluate(w=w, n_flows=n_flows, n_packets=n_packets)

        results.append({
            "w": w,
            "std_mean": err_std.mean(),
            "var_mean": err_var.mean(),
            "std_max": err_std.max(),
            "var_max": err_var.max(),
        })

    print("\n========== RESULTS ==========")
    for r in results:
        print(
            f"w={r['w']:6d} | "
            f"STD mean={r['std_mean']:.3f}, VAR mean={r['var_mean']:.3f} | "
            f"STD max={r['std_max']}, VAR max={r['var_max']}"
        )

    return results


# ================================================================
# MAIN
# ================================================================
if __name__ == "__main__":
    widths = [2000, 5000, 10000, 20000, 50000]
    sweep_widths(widths)
