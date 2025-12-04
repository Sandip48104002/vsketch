import random
from collections import Counter
import xxhash



# ==============================================================================
#  STANDARD COUNT-MIN SKETCH (Baseline)
# ==============================================================================

class CMS_Standard:

    def __init__(self, rows=4, width=20000):
        self.rows = rows
        self.width = width
        self.C = [[0] * width for _ in range(rows)]
        self.row_seeds = [random.randint(1, 10**9) for _ in range(rows)]

    def h_row(self, j: int, key: str):
        return xxhash.xxh32(key, seed=self.row_seeds[j]).intdigest() % self.width

    def insert(self, ip: str):
        for j in range(self.rows):
            col = self.h_row(j, ip)
            self.C[j][col] += 1

    def query(self, ip: str):
        return min(self.C[j][self.h_row(j, ip)] for j in range(self.rows))



# ==============================================================================
#  VARIANT COUNT-MIN SKETCH  (Your new system)
# ==============================================================================

class CMS_Variant:

    def __init__(self, rows=4, width=20000, k=3, conservative=True):
        self.rows = rows
        self.width = width
        self.k = k
        self.conservative = conservative

        self.s = 0                     # global packet counter
        self.C = [[0] * width for _ in range(rows)]
        self.row_seeds = [random.randint(1, 10**9) for _ in range(rows)]

    def H0(self, ip: str) -> int:
        return xxhash.xxh32(ip).intdigest()

    def h_row(self, j: int, key: str):
        return xxhash.xxh32(key, seed=self.row_seeds[j]).intdigest() % self.width

    def insert(self, ip: str):
        self.s += 1

        v = (self.H0(ip) + self.s) % self.k
        key = f"{ip}:{v}"

        cols = [self.h_row(j, key) for j in range(self.rows)]
        if self.conservative:
            # increment only the *minimum* counter
            vals = [self.C[j][cols[j]] for j in range(self.rows)]
            min_val = min(vals)
            for j in range(self.rows):
                if self.C[j][cols[j]] == min_val:
                    self.C[j][cols[j]] += 1
        else:
            # normal CMS update
            for j in range(self.rows):
                self.C[j][cols[j]] += 1

    def query(self, ip: str):
        total = 0
        for v in range(self.k):
            key = f"{ip}:{v}"
            total += min(self.C[j][self.h_row(j, key)] for j in range(self.rows))
        return total



# ==============================================================================
#  TEST HARNESS
# ==============================================================================

def main():

    random.seed(12345)

    ROWS = 4
    WIDTH = 20000
    K = 3

    cms_std = CMS_Standard(rows=ROWS, width=WIDTH)
    cms_var = CMS_Variant(rows=ROWS, width=WIDTH, k=K, conservative=True)

    # --- Traffic Model ---
    IPS_HEAVY  = [f"10.0.0.{i}" for i in range(1, 11)]
    IPS_MEDIUM = [f"192.168.1.{i}" for i in range(1, 51)]
    IPS_LIGHT  = [f"172.16.0.{i}" for i in range(1, 5001)]

    TRUE = Counter()

    TOTAL = 300_000

    print(f"Generating {TOTAL} packets...")

    for _ in range(TOTAL):
        r = random.random()
        if r < 0.5:
            ip = random.choice(IPS_HEAVY)
        elif r < 0.8:
            ip = random.choice(IPS_MEDIUM)
        else:
            ip = random.choice(IPS_LIGHT)

        TRUE[ip] += 1
        cms_std.insert(ip)
        cms_var.insert(ip)

    # ==================================================================
    #  HEAVY HITTER OUTPUT
    # ==================================================================
    # print("\n========== HEAVY HITTERS ==========")
    # print("IP               TRUE     STD CMS     VAR CMS")
    # for ip in IPS_HEAVY:
    #     t = TRUE[ip]
    #     a = cms_std.query(ip)
    #     b = cms_var.query(ip)
    #     print(f"{ip:12s}   {t:7d}   {a:10d}   {b:10d}")

    # ==================================================================
    #  SAMPLE MEDIUM
    # ==================================================================
    # print("\n========== MEDIUM HITTERS (sample) ==========")
    # for ip in random.sample(IPS_MEDIUM, 10):
    #     print(f"{ip:12s}  true={TRUE[ip]:6d}  std={cms_std.query(ip):6d}  var={cms_var.query(ip):6d}")

    # ==================================================================
    #  SAMPLE LIGHT
    # ==================================================================
    # print("\n========== LIGHT IPs (sample) ==========")
    # for ip in random.sample(IPS_LIGHT, 10):
    #     print(f"{ip:12s}  true={TRUE[ip]:6d}  std={cms_std.query(ip):6d}  var={cms_var.query(ip):6d}")

    # ==================================================================
    #  ERROR ANALYSIS
    # ==================================================================
    std_abs = []
    var_abs = []

    for ip in TRUE:
        t = TRUE[ip]
        std_abs.append(cms_std.query(ip) - t)
        var_abs.append(cms_var.query(ip) - t)

    print("\n========== ERROR STATISTICS ==========")
    print("\n--- Standard CMS ---")
    print("Mean abs error: ", sum(std_abs) / len(std_abs))
    print("Max  abs error: ", max(std_abs))
    print("Min  abs error: ", min(std_abs))

    print("\n--- Variant CMS ---")
    print("Mean abs error: ", sum(var_abs) / len(var_abs))
    print("Max  abs error: ", max(var_abs))
    print("Min  abs error: ", min(var_abs))

    print("\n--- Improvement ---")
    improvement = (sum(std_abs) - sum(var_abs)) / sum(std_abs) * 100
    print(f"Variant CMS reduces total overestimation by: {improvement:.2f}%")

    # ==============================================================================
    #  DIFFERENCE REPORT
    # ==============================================================================

    # print("\n========== IPS WITH DIFFERENCES (STD ≠ VAR) ==========")

    diff_ips = []

    for ip in TRUE:
        std_val = cms_std.query(ip)
        var_val = cms_var.query(ip)
        if std_val != var_val:
            diff_ips.append((ip, TRUE[ip], std_val, var_val))

    # if not diff_ips:
    #     print("No differences found.")
    # else:
    #     for ip, t, s, v in diff_ips:
            # print(f"{ip:15s}  true={t:6d}  std={s:6d}  var={v:6d}")

    # print(f"\nTotal IPs with differences: {len(diff_ips)}")

    # ==============================================================================
    #  EXPORT DIFFERENCES TO CSV
    # ==============================================================================

    import csv

    csv_file = "cms_differences.csv"

    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["IP", "TRUE_COUNT", "STD_CMS", "VARIANT_CMS", "ABS_ERROR_STD", "ABS_ERROR_VAR"])

        for ip, t, s, v in diff_ips:
            writer.writerow([ip, t, s, v, s - t, v - t])

    print(f"\nCSV file written: {csv_file}")



# ==============================================================================
if __name__ == "__main__":
    main()
