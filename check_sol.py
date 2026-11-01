#!/usr/bin/env python3
"""Independent checker: python3 check_sol.py instance.vrp solution.sol [round=1]
Parses the .vrp, reads 'Route #k: ...' lines (depot = 0, customers 1..n-1),
checks every customer is served exactly once and no route exceeds capacity,
and recomputes the cost (rounded Euclidean if round=1)."""
import math, sys
def read_vrp(fn):
    lines = open(fn).read().splitlines(); n = cap = 0; xy = []; dem = []; i = 0
    while i < len(lines):
        l = lines[i]
        if l.startswith("DIMENSION"): n = int(l.split(":")[1])
        elif l.startswith("CAPACITY"): cap = float(l.split(":")[1])
        elif l.startswith("NODE_COORD_SECTION"):
            xy = [tuple(map(float, lines[i + 1 + k].split()[1:3])) for k in range(n)]; i += n
        elif l.startswith("DEMAND_SECTION"):
            dem = [float(lines[i + 1 + k].split()[1]) for k in range(n)]; i += n
        i += 1
    return n, cap, xy, dem
inst, sol = sys.argv[1], sys.argv[2]; rnd = int(sys.argv[3]) if len(sys.argv) > 3 else 1
n, cap, xy, dem = read_vrp(inst)
def d(a, b):
    w = math.hypot(xy[a][0] - xy[b][0], xy[a][1] - xy[b][1]); return float(round(w)) if rnd else w
routes = [list(map(int, l.split(":")[1].split())) for l in open(sol) if l.startswith("Route")]
seen = [0] * n; cost = 0.0; ok = True
for r in routes:
    if sum(dem[v] for v in r) > cap + 1e-9: ok = False; print("capacity violated", r[:5])
    for v in r: seen[v] += 1
    path = [0] + r + [0]; cost += sum(d(path[k], path[k + 1]) for k in range(len(path) - 1))
if any(seen[v] != 1 for v in range(1, n)) or seen[0]: ok = False; print("customer coverage wrong")
claimed = [float(l.split()[1]) for l in open(sol) if l.startswith("Cost")][0]
print(f"{inst.split('/')[-1]}: routes={len(routes)} cost={cost:.2f} claimed={claimed:.2f} "
      f"{'OK' if ok and abs(cost - claimed) < 0.01 else 'MISMATCH/INVALID'}")
