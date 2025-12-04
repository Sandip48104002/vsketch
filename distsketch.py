# coding: utf-8
import xxhash
import random
from collections import Counter
# ==============================================================================
class CountMinSketch:
    def __init__(self, rows, width, k):
        self.rows = rows
        self.width = width
        self.k = k                      # number of variants
        self.C = [[0] * width for _ in range(rows)]
        self.s = 0                      # global packet counter

        # row-wise hash seeds for independent hash functions
        self.row_seeds = [i * 1234567 + 987654321 for i in range(rows)]

    # 32-bit hash for IP only
    def H0(self, key: str) -> int:
        return xxhash.xxh32(key).intdigest()

    # CMS row hash function
    def H_row(self, row: int, key: str) -> int:
        return xxhash.xxh32(key, seed=self.row_seeds[row]).intdigest() % self.width

    # Insert packet
    def insert(self, ip: str):
        self.s += 1                      # global counter increment
        v = (self.H0(ip) + self.s) % self.k    # variant selection
        key = f"{ip}:{v}"                # transformed key

        for r in range(self.rows):
            col = self.H_row(r, key)
            self.C[r][col] += 1

    # Query original IP
    def query(self, ip: str) -> int:
        total = 0
        for v in range(self.k):
            key = f"{ip}:{v}"

            # CMS estimate = min across rows
            est = float("inf")
            for r in range(self.rows):
                col = self.H_row(r, key)
                est = min(est, self.C[r][col])

            total += est
        return total

cms = CountMinSketch(rows=4, width=5000, k=3)
TRUE = Counter()

IPS_HEAVY  = [f"10.0.0.{i}" for i in range(1, 11)]
IPS_MEDIUM = [f"192.168.1.{i}" for i in range(1, 51)]
IPS_RANDOM = [f"172.16.0.{i}" for i in range(1, 5001)]

print("Total heavy IPs:  ", len(IPS_HEAVY))
print("Total medium IPs: ", len(IPS_MEDIUM))
print("Total random IPs: ", len(IPS_RANDOM))

N = 300_000
print("Generating packets...")

generated_sample = []

for _ in range(N):
    r = random.random()

    if r < 0.5:
        ip = random.choice(IPS_HEAVY)   # 50%
    elif r < 0.8:
        ip = random.choice(IPS_MEDIUM)  # 30%
    else:
        ip = random.choice(IPS_RANDOM)  # 20%

    TRUE[ip] += 1
    cms.insert(ip)

    if random.random() < 0.00001:
        generated_sample.append(ip)

print("Sample generated packets:", generated_sample[:20])
print("\n===== Heavy Hitters =====")
for ip in IPS_HEAVY:
    print(f"{ip:12s}   true={TRUE[ip]:8d}   est={cms.query(ip):8d}")

# --- Print Medium results ------------------------------------------------------

print("\n===== Medium Hitters =====")
for ip in random.sample(IPS_MEDIUM, 10):
    print(f"{ip:12s}   true={TRUE[ip]:8d}   est={cms.query(ip):8d}")

# --- Print Random results ------------------------------------------------------

print("\n===== Random Light IPs =====")
for ip in random.sample(IPS_RANDOM, 10):
    print(f"{ip:12s}   true={TRUE[ip]:8d}   est={cms.query(ip):8d}")
absolute_errors = []
relative_errors = []

for ip in TRUE:
    true_val = TRUE[ip]
    est_val = cms.query(ip)
    abs_err = est_val - true_val

    absolute_errors.append(abs_err)
    if true_val > 0:
        relative_errors.append(abs_err / true_val)

print("\n===== ERROR STATISTICS =====")
print("Mean absolute error:      ", sum(absolute_errors) / len(absolute_errors))
print("Mean relative error (%):  ", 100 * sum(relative_errors) / len(relative_errors))
print("Max absolute error:       ", max(absolute_errors))
print("Min absolute error:       ", min(absolute_errors))
