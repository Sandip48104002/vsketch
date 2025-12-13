import random
import xxhash
import numpy as np

# ==============================================================================
#  STANDARD COUNT-MIN SKETCH
# ==============================================================================
class CMS_Standard:
    def __init__(self, rows=4, width=20000):
        self.rows = rows
        self.width = width
        self.C = [[0]*width for _ in range(rows)]
        self.row_seeds = [random.randint(1, 1_000_000_000) for _ in range(rows)]

    def h_row(self, j, key):
        return xxhash.xxh32(key, seed=self.row_seeds[j]).intdigest() % self.width

    def insert(self, key):
        for j in range(self.rows):
            col = self.h_row(j, key)
            self.C[j][col] += 1

    def query(self, key):
        return min(self.C[j][self.h_row(j, key)] for j in range(self.rows))

    # return % utilization
    def utilization(self):
        return np.count_nonzero(np.array(self.C)) / (self.rows * self.width)

    # return (avg count, max count, empty buckets per row)
    def stats(self):
        C = np.array(self.C)

        row_avgs = []
        row_max = []
        row_zeros = []

        for r in range(self.rows):
            row = C[r]
            nonzero = row[row > 0]

            if len(nonzero) == 0:
                avg_nonzero = 0.0
            else:
                avg_nonzero = nonzero.sum() / len(nonzero)

            row_avgs.append(avg_nonzero)
            row_max.append(row.max())
            row_zeros.append(np.sum(row == 0))

        return row_avgs, row_max, row_zeros


# ==============================================================================
#  VARIANT COUNT-MIN SKETCH
# ==============================================================================
class CMS_Variant:
    def __init__(self, rows=4, width=20000, k=3, conservative=False):
        self.rows = rows
        self.width = width
        self.k = k
        self.conservative = conservative
        self.s = 0
        self.C = [[0]*width for _ in range(rows)]
        self.row_seeds = [random.randint(1, 1_000_000_000) for _ in range(rows)]

    def H0(self, key):
        return xxhash.xxh32(key).intdigest()

    def h_row(self, j, key):
        return xxhash.xxh32(key, seed=self.row_seeds[j]).intdigest() % self.width

    def insert(self, key):
        self.s += 1
        v = (self.H0(key) + self.s) % self.k
        variant_key = f"{key}:{v}"
        cols = [self.h_row(j, variant_key) for j in range(self.rows)]

        if self.conservative:
            vals = [self.C[j][cols[j]] for j in range(self.rows)]
            min_val = min(vals)
            for j in range(self.rows):
                if self.C[j][cols[j]] == min_val:
                    self.C[j][cols[j]] += 1
        else:
            for j in range(self.rows):
                self.C[j][cols[j]] += 1

    def query(self, key):
        total = 0
        for v in range(self.k):
            variant_key = f"{key}:{v}"
            total += min(self.C[j][self.h_row(j, variant_key)] for j in range(self.rows))
        return total

    def utilization(self):
        return np.count_nonzero(np.array(self.C)) / (self.rows * self.width)

    # return (avg count, max count, empty buckets per row)
    def stats(self):
        C = np.array(self.C)

        row_avgs = []
        row_max = []
        row_zeros = []

        for r in range(self.rows):
            row = C[r]
            nonzero = row[row > 0]

            if len(nonzero) == 0:
                avg_nonzero = 0.0
            else:
                avg_nonzero = nonzero.sum() / len(nonzero)

            row_avgs.append(avg_nonzero)
            row_max.append(row.max())
            row_zeros.append(np.sum(row == 0))

        return row_avgs, row_max, row_zeros


# ==============================================================================
# TRAFFIC GENERATION
# ==============================================================================
def generate_flows(n_flows, n_packets, zipf_param=1.2):
    flows = [f"IP{i}" for i in range(n_flows)]

    weights = np.random.zipf(zipf_param, n_flows).astype(np.float64)
    weights = np.clip(weights, 0, None)
    weights /= weights.sum()
    weights[-1] = 1.0 - weights[:-1].sum()
    weights = np.clip(weights, 0, None)
    weights /= weights.sum()

    packets = np.random.choice(flows, size=n_packets, p=weights)
    true_counts = {f: 0 for f in flows}
    for p in packets:
        true_counts[p] += 1

    return packets, true_counts


# ==============================================================================
# EVALUATION
# ==============================================================================
def evaluate(n_flows=10000, n_packets=1000000, rows=3, width=20000, k=5):
    packets, TRUE = generate_flows(n_flows, n_packets)

    cms_std = CMS_Standard(rows, width)
    cms_var = CMS_Variant(rows, width, k)

    for p in packets:
        cms_std.insert(p)
        cms_var.insert(p)

    errors_std = [cms_std.query(ip)-TRUE[ip] for ip in TRUE]
    errors_var = [cms_var.query(ip)-TRUE[ip] for ip in TRUE]

    util_std = cms_std.utilization()
    util_var = cms_var.utilization()

    stats_std = cms_std.stats()
    stats_var = cms_var.stats()

    return errors_std, errors_var, util_std, util_var, stats_std, stats_var


# ==============================================================================
# SWEEP
# ==============================================================================
def sweep_widths():
    widths = [2000, 5000, 10000]

    print(f"{'W':>6} | {'STD mean':>8} | {'VAR mean':>8} | {'STD util':>8} | {'VAR util':>8}")
    print("-"*60)
    for w in widths:
        errors_std, errors_var, util_std, util_var, stats_std, stats_var = evaluate(width=w)

        print(f"{w:6} | {np.mean(errors_std):8.3f} | {np.mean(errors_var):8.3f} | "
              f"{util_std*100:8.2f}% | {util_var*100:8.2f}%")

        row_avg_std, row_max_std, row_zero_std = stats_std
        row_avg_var, row_max_var, row_zero_var = stats_var

        print("\n--- Standard CMS Stats ---")
        for i in range(len(row_avg_std)):
            print(f"Row {i}: avg={row_avg_std[i]:.3f}, max={row_max_std[i]}, empty={row_zero_std[i]}")

        print("\n--- Variant CMS Stats ---")
        for i in range(len(row_avg_var)):
            print(f"Row {i}: avg={row_avg_var[i]:.3f}, max={row_max_var[i]}, empty={row_zero_var[i]}")
        print("\n" + "-"*60)


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    sweep_widths()
