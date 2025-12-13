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
        self.counter_keys = [[set() for _ in range(width)] for _ in range(rows)]
        self.row_seeds = [random.randint(1, 1_000_000_000) for _ in range(rows)]
        self.collisions = 0

    def h_row(self, j, key):
        return xxhash.xxh32(key, seed=self.row_seeds[j]).intdigest() % self.width

    def insert(self, key):
        for j in range(self.rows):
            col = self.h_row(j, key)
            # True collision: key not seen before but counter non-zero
            if key not in self.counter_keys[j][col] and self.C[j][col] != 0:
                self.collisions += 1
            self.counter_keys[j][col].add(key)
            self.C[j][col] += 1

    def query(self, key):
        return min(self.C[j][self.h_row(j, key)] for j in range(self.rows))

    def utilization(self):
        return np.count_nonzero(np.array(self.C)) / (self.rows*self.width)
    
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
        self.C = [[0]*width for _ in range(rows)]
        self.counter_keys = [[set() for _ in range(width)] for _ in range(rows)]
        self.row_seeds = [random.randint(1, 1_000_000_000) for _ in range(rows)]
        self.collisions = 0
        self.s = 0  # global packet counter

    def H0(self, key):
        return xxhash.xxh32(key).intdigest()

    def h_row(self, j, key):
        return xxhash.xxh32(key, seed=self.row_seeds[j]).intdigest() % self.width

    def insert(self, key):
        self.s += 1
        v = (self.H0(key) + self.s) % self.k
        variant_key = f"{key}:{v}"
        cols = [self.h_row(j, variant_key) for j in range(self.rows)]

        # if self.conservative:
        #     vals = [self.C[j][cols[j]] for j in range(self.rows)]
        #     min_val = min(vals)
        #     for j in range(self.rows):
        #         if self.C[j][cols[j]] == min_val:
        #             # True collision check
        #             if variant_key not in self.counter_keys[j][cols[j]] and self.C[j][cols[j]] != 0:
        #                 self.collisions += 1
        #             self.counter_keys[j][cols[j]].add(variant_key)
        #             self.C[j][cols[j]] += 1
        # else:
        for j in range(self.rows):
            if variant_key not in self.counter_keys[j][cols[j]] and self.C[j][cols[j]] != 0:
                self.collisions += 1
            self.counter_keys[j][cols[j]].add(variant_key)
            self.C[j][cols[j]] += 1

    def query(self, key):
        total = 0
        for v in range(self.k):
            variant_key = f"{key}:{v}"
            total += min(self.C[j][self.h_row(j, variant_key)] for j in range(self.rows))
        return total

    def utilization(self):
        return np.count_nonzero(np.array(self.C)) / (self.rows*self.width)
    
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
# Traffic generation
# ==============================================================================
# Convert 32-bit integer to IPv4 dotted string
def _int_to_ip(value: int) -> str:
    value &= 0xFFFFFFFF
    return f"{(value >> 24) & 0xFF}.{(value >> 16) & 0xFF}.{(value >> 8) & 0xFF}.{value & 0xFF}"


def generate_traffic(n_flows, n_packets, zipf_param=1.2, seed=42):
    """Create packet stream with flow metadata.

    Returns a dict containing:
        - packets_ids: numpy array of flow IDs per packet
        - packets_ips: numpy array of IP strings per packet
        - flow_table: list of {flow_id, ip}
        - true_counts_ids: dict[flow_id] -> packet count
        - true_counts_ips: dict[ip] -> packet count
    """

    if n_packets < n_flows:
        raise ValueError("n_packets must be >= n_flows to guarantee at least one packet per flow.")

    rng = np.random.default_rng(seed)

    flow_ids = np.arange(n_flows, dtype=np.int64)

    # Sample unique IPv4 addresses without replacement across the full 2^32 space
    ip_values = []
    seen = set()
    while len(ip_values) < n_flows:
        needed = n_flows - len(ip_values)
        samples = rng.integers(0, 2**32, size=needed * 2, dtype=np.uint64)
        for val in samples:
            ip_int = int(val)
            if ip_int in seen:
                continue
            seen.add(ip_int)
            ip_values.append(ip_int)
            if len(ip_values) == n_flows:
                break

    flow_ips = np.array([_int_to_ip(ip_int) for ip_int in ip_values], dtype=object)

    weights = rng.zipf(zipf_param, n_flows).astype(np.float64)
    weights = np.clip(weights, 0, None)
    weights /= weights.sum()
    weights[-1] = 1.0 - np.sum(weights[:-1])
    weights = np.clip(weights, 0, None)
    weights /= weights.sum()

    packets_ids = np.empty(n_packets, dtype=np.int64)
    packets_ids[:n_flows] = flow_ids

    remaining = n_packets - n_flows
    unique_sampled = np.array([], dtype=np.int64)
    counts = np.array([], dtype=np.int64)
    if remaining > 0:
        sampled_ids = rng.choice(flow_ids, size=remaining, p=weights)
        packets_ids[n_flows:] = sampled_ids
        unique_sampled, counts = np.unique(sampled_ids, return_counts=True)

    rng.shuffle(packets_ids)

    true_counts_ids = np.ones(n_flows, dtype=np.int64)
    if unique_sampled.size:
        true_counts_ids[unique_sampled] += counts

    packets_ips = flow_ips[packets_ids]

    true_counts_ids_map = {int(flow_id): int(true_counts_ids[flow_id]) for flow_id in flow_ids}
    true_counts_ips = {str(flow_ips[flow_id]): int(true_counts_ids[flow_id]) for flow_id in flow_ids}

    flow_table = [
        {"flow_id": int(flow_id), "ip": str(flow_ips[flow_id])}
        for flow_id in flow_ids
    ]

    return {
        "packets_ids": packets_ids,
        "packets_ips": packets_ips,
        "flow_table": flow_table,
        "true_counts_ids": true_counts_ids_map,
        "true_counts_ips": true_counts_ips,
    }


# Generate flows with random IPv4 addresses and associated IDs
def generate_flows(n_flows, n_packets, zipf_param=1.2, seed=42):
    traffic = generate_traffic(n_flows, n_packets, zipf_param=zipf_param, seed=seed)
    return (
        traffic["packets_ips"],
        traffic["true_counts_ips"],
        traffic["packets_ids"],
        traffic["true_counts_ids"],
        traffic["flow_table"],
    )

# ==============================================================================
# Evaluation
# ==============================================================================
def evaluate(
    n_flows=10000,
    n_packets=30000000,
    rows=4,
    width=4096,
    k=5,
    seed=42,
    key_mode="ip",
    traffic=None,
):
    if traffic is None:
        traffic = generate_flows(n_flows, n_packets, seed=seed)

    packets_ips, true_counts_ip, packet_ids, true_counts_ids, flow_table = traffic

    if key_mode == "ip":
        packet_keys = packets_ips
        true_counts = true_counts_ip
        key_desc = "IP"
    elif key_mode in ("flow_id", "id"):
        packet_keys = np.array([str(int(fid)) for fid in packet_ids], dtype=object)
        true_counts = {str(int(fid)): int(count) for fid, count in true_counts_ids.items()}
        key_desc = "Flow ID"
    else:
        raise ValueError(f"Unsupported key_mode: {key_mode}")

    sample_items = list(true_counts.items())[:10]
    # print(f"True-count sample ({len(true_counts)} flows total) using {key_desc} keys: {sample_items}")

    cms_std = CMS_Standard(rows, width)
    cms_var = CMS_Variant(rows, width, k)

    for key in packet_keys:
        cms_std.insert(key)
        cms_var.insert(key)

    errors_std = [cms_std.query(key) - true_counts[key] for key in true_counts]
    errors_var = [cms_var.query(key) - true_counts[key] for key in true_counts]

    util_std = cms_std.utilization()
    util_var = cms_var.utilization()

    coll_std = cms_std.collisions
    coll_var = cms_var.collisions

    stats_std = cms_std.stats()
    stats_var = cms_var.stats()

    return {
        "errors_std": errors_std,
        "errors_var": errors_var,
        "util_std": util_std,
        "util_var": util_var,
        "coll_std": coll_std,
        "coll_var": coll_var,
        "stats_std": stats_std,
        "stats_var": stats_var,
        "packets_ids": packet_ids,
        "true_counts_ids": true_counts_ids,
        "flow_table": flow_table,
        "key_mode": key_mode,
        "key_desc": key_desc,
    }

# ==============================================================================
# Run sweep over widths
# ==============================================================================
def sweep_widths(seed=42):
    widths = [16384]
    print("\nUsing Flow Generation")

    print("number of flows: 10,000; number of packets: 30,000,000; rows: 4; k: 3")
    print(f"{'Width':>6} | {'CMS mean':>8} | {'VAR mean':>8} | {'CMS util':>8} | {'VAR util':>8} | {'Coll CMS':>9} | {'Coll VAR':>9}")
    for w in widths:
        traffic = generate_flows(10000, 30000000, seed=seed)
        eval_ip = evaluate(width=w, seed=seed, traffic=traffic, key_mode="ip")
        eval_id = evaluate(width=w, seed=seed, traffic=traffic, key_mode="flow_id")

        def summarize(result):
            errors_std = result["errors_std"]
            errors_var = result["errors_var"]
            util_std = result["util_std"]
            util_var = result["util_var"]
            coll_std = result["coll_std"]
            coll_var = result["coll_var"]
            stats_std = result["stats_std"]
            stats_var = result["stats_var"]

            row_avg_std, row_max_std, row_zero_std = stats_std
            row_avg_var, row_max_var, row_zero_var = stats_var

            mean_row_avg_std = np.mean(row_avg_std)
            mean_row_max_std = np.mean(row_max_std)
            mean_row_zero_std = np.mean(row_zero_std)

            mean_row_avg_var = np.mean(row_avg_var)
            mean_row_max_var = np.mean(row_max_var)
            mean_row_zero_var = np.mean(row_zero_var)

            avg_reduction = ((mean_row_avg_std - mean_row_avg_var) / mean_row_avg_std * 100.0) if mean_row_avg_std else 0.0
            max_reduction = ((mean_row_max_std - mean_row_max_var) / mean_row_max_std * 100.0) if mean_row_max_std else 0.0
            zero_reduction = ((mean_row_zero_std - mean_row_zero_var) / mean_row_zero_std * 100.0) if mean_row_zero_std else 0.0

            return {
                "mean_err_std": np.mean(errors_std),
                "mean_err_var": np.mean(errors_var),
                "util_std": util_std,
                "util_var": util_var,
                "coll_std": coll_std,
                "coll_var": coll_var,
                "mean_row_avg_std": mean_row_avg_std,
                "mean_row_max_std": mean_row_max_std,
                "mean_row_zero_std": mean_row_zero_std,
                "mean_row_avg_var": mean_row_avg_var,
                "mean_row_max_var": mean_row_max_var,
                "mean_row_zero_var": mean_row_zero_var,
                "avg_reduction": avg_reduction,
                "max_reduction": max_reduction,
                "zero_reduction": zero_reduction,
            }

        summary_ip = summarize(eval_ip)
        summary_id = summarize(eval_id)

        print(f"          IP | {summary_ip['mean_err_std']:8.3f} | {summary_ip['mean_err_var']:8.3f} | {summary_ip['util_std']*100:8.2f}% | {summary_ip['util_var']*100:8.2f}% | {summary_ip['coll_std']:9d} | {summary_ip['coll_var']:9d} | (keys=IP)")
        print(f"      FlowID | {summary_id['mean_err_std']:8.3f} | {summary_id['mean_err_var']:8.3f} | {summary_id['util_std']*100:8.2f}% | {summary_id['util_var']*100:8.2f}% | {summary_id['coll_std']:9d} | {summary_id['coll_var']:9d} |")

        print("      CMS/IP : rows avg={:.3f}, max_value={:.3f}, empty_bkt={:.1f}".format(
            summary_ip["mean_row_avg_std"],
            summary_ip["mean_row_max_std"],
            summary_ip["mean_row_zero_std"],
        ))
        print("      CMS/ID : rows avg={:.3f}, max_value={:.3f}, empty_bkt={:.1f}".format(
            summary_id["mean_row_avg_std"],
            summary_id["mean_row_max_std"],
            summary_id["mean_row_zero_std"],
        ))
        print("      VAR/IP : rows avg={:.3f}, max_value={:.3f}, empty_bkt={:.1f}".format(
            summary_ip["mean_row_avg_var"],
            summary_ip["mean_row_max_var"],
            summary_ip["mean_row_zero_var"],
        ))
        print("      VAR/ID : rows avg={:.3f}, max_value={:.3f}, empty_bkt={:.1f}".format(
            summary_id["mean_row_avg_var"],
            summary_id["mean_row_max_var"],
            summary_id["mean_row_zero_var"],
        ))

        print("      Reduction/IP: avg={:.2f}%, max_value={:.2f}%, empty_bkt={:.2f}%".format(
            summary_ip["avg_reduction"],
            summary_ip["max_reduction"],
            summary_ip["zero_reduction"],
        ))
        print("      Reduction/ID: avg={:.2f}%, max_value={:.2f}%, empty_bkt={:.2f}%".format(
            summary_id["avg_reduction"],
            summary_id["max_reduction"],
            summary_id["zero_reduction"],
        ))

        print("      Delta(FlowID - IP): util_std={:+.2f}pp, util_var={:+.2f}pp, coll_std={:+d}, coll_var={:+d}".format(
            (summary_id["util_std"] - summary_ip["util_std"]) * 100,
            (summary_id["util_var"] - summary_ip["util_var"]) * 100,
            summary_id["coll_std"] - summary_ip["coll_std"],
            summary_id["coll_var"] - summary_ip["coll_var"],
        ))

# ==============================================================================
# Main
# ==============================================================================
if __name__ == "__main__":
    # sweep_widths(1)
    sweep_widths()
