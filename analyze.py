#!/usr/bin/env python3
"""Summarise results/*.csv into results/summary.md (times, costs, gaps to BKS)."""
import csv, math, os, re, statistics as st
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
BKS = dict(l.split() for l in open(os.path.join(HERE, "bks.txt")))
BKS = {k: float(v) for k, v in BKS.items()}
ORDER = [l.strip() for l in open(os.path.join(HERE, "all.txt")) if l.strip()]


def parse(line):
    """-> dict(costs=[...], times=[...], valid=bool) or None"""
    if "Cost" not in line or "Time(seconds)" not in line:
        return None
    head, tail = line.split("Time(seconds)")
    costs = [float(x) for x in re.findall(r"[-+]?\d+\.?\d*(?:e[-+]?\d+)?", head.split("Cost", 1)[1])]
    nums = tail.split()
    times = []
    for x in nums:
        try:
            times.append(float(x))
        except ValueError:
            break
    valid = line.rstrip().endswith("VALID") and not line.rstrip().endswith("INVALID")
    return dict(costs=costs, times=times, valid=valid)


def load(fname):
    """-> {label: {instance: parsed}}"""
    out = defaultdict(dict)
    p = os.path.join(R, fname)
    if not os.path.exists(p):
        return out
    for row in csv.reader(open(p)):
        if len(row) < 3:
            continue
        label, inst, line = row[0], row[1], ",".join(row[2:])
        out[label][inst] = parse(line)
    return out


def gap(c, inst):
    return 100.0 * (c - BKS[inst]) / BKS[inst]


def gmean(xs):
    xs = [x for x in xs if x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


lines = []
P = lines.append

# ---------------- E1: original vs C port ----------------
o = load("orig_seq.csv").get("orig_seq", {})
cs = load("c_seq.csv").get("c_seq", {})
cp = load("c_par2.csv").get("c_par2", {})
ls = load("c_ls.csv").get("c_ls", {})
il = load("c_ils10.csv").get("c_ils10", {})

common = [i for i in ORDER if o.get(i) and cs.get(i) and cp.get(i)]
P("# Results summary\n")
P("Machine: 2-core Intel Xeon @ 2.80 GHz (AVX-512), 7 GB RAM, gcc 13.3, -O3 -march=native.\n")
P("## E1. Original C++ code vs the C port (same algorithm, `-ls 0`)\n")
if common:
    to = sum(o[i]["times"][-1] for i in common)
    t1 = sum(cs[i]["times"][-1] for i in common)
    t2 = sum(cp[i]["times"][-1] for i in common)
    P(f"Instances where all three ran: **{len(common)}** "
      f"(original code ran out of memory on: {', '.join(i for i in ORDER if not (o.get(i)))}).\n")
    P("| Version | Total time (s) | Speed-up vs original seqMDS | Avg gap to BKS (%) | Geo-mean gap (%) | All valid |")
    P("|---|---:|---:|---:|---:|:-:|")
    for name, d, t in [("Original seqMDS (C++, 1 thread)", o, to), ("C port, 1 thread", cs, t1), ("C port, 2 threads (OpenMP)", cp, t2)]:
        g = [gap(d[i]["costs"][-1], i) for i in common]
        P(f"| {name} | {t:.1f} | {to / t:.2f}x | {st.mean(g):.2f} | {gmean(g):.2f} | {'yes' if all(d[i]['valid'] for i in common) else 'NO'} |")
    P("")
    # per-family
    fam = defaultdict(list)
    for i in common:
        fam[re.split(r"[-_0-9]", i)[0] or i].append(i)
    P("Per family (total seconds):\n")
    P("| Family | n inst | orig seq | C 1 thr | C 2 thr |")
    P("|---|---:|---:|---:|---:|")
    for f, ins in sorted(fam.items()):
        P(f"| {f} | {len(ins)} | {sum(o[i]['times'][-1] for i in ins):.1f} | {sum(cs[i]['times'][-1] for i in ins):.1f} | {sum(cp[i]['times'][-1] for i in ins):.1f} |")
    P("")
    extra = [i for i in ORDER if cs.get(i) and cp.get(i) and i not in common]
    if extra:
        P("Instances only the C port could run (original needs an n×n edge list):\n")
        P("| Instance | C 1 thr (s) | C 2 thr (s) | cost |")
        P("|---|---:|---:|---:|")
        for i in extra:
            P(f"| {i} | {cs[i]['times'][-1]:.1f} | {cp[i]['times'][-1]:.1f} | {cp[i]['costs'][-1]:.0f} |")
        P("")

# ---------------- E2: schedules ----------------
sc = load("sched.csv")
sub = [l.strip() for l in open(os.path.join(HERE, "subset.txt")) if l.strip()]
if sc:
    P("## E2. Parallel runtime / schedule experiments (2 threads, `-ls 0`, rho = 1e5)\n")
    P("Time of the randomized phase + post-processing (seconds). Costs are identical in every row "
      "(each iteration is seeded by its index), so only time changes.\n")
    labels = list(sc.keys())
    P("| Config | " + " | ".join(sub) + " | Total |")
    P("|---|" + "---:|" * (len(sub) + 1))
    for lab in labels:
        row = [sc[lab].get(i) for i in sub]
        ts = [r["times"][-1] if r else float("nan") for r in row]
        P(f"| {lab} | " + " | ".join(f"{t:.2f}" for t in ts) + f" | {sum(t for t in ts if t == t):.2f} |")
    P("")
    # check identical costs
    costs = {i: set(sc[l][i]["costs"][-1] for l in labels if sc[l].get(i)) for i in sub}
    P("Distinct final costs per instance across configs: " +
      ", ".join(f"{i}: {len(c)}" for i, c in costs.items()) + "\n")

simd = load("simd.csv")
if simd:
    P("## SIMD: Prim's MST time (`-iters 1`, 2 threads), seconds\n")
    P("| Build | " + " | ".join(sub) + " |")
    P("|---|" + "---:|" * len(sub))
    for lab in simd:
        P(f"| {lab} | " + " | ".join(f"{simd[lab][i]['times'][0]:.4f}" if simd[lab].get(i) else "-" for i in sub) + " |")
    P("")

op = load("orig_par2.csv").get("orig_par2", {})
if op:
    P("## Original parMDS compiled with g++ -fopenmp (2 threads)\n")
    P("| Instance | reported cost | time (s) | verdict |")
    P("|---|---:|---:|---|")
    for i in sub:
        r = op.get(i)
        if r:
            P(f"| {i} | {r['costs'][-1]:.0f} | {r['times'][-1]:.2f} | {'VALID' if r['valid'] else 'INVALID'} |")
        else:
            P(f"| {i} | - | - | failed / out of memory |")
    P("")

# ---------------- E3/E4: quality ----------------
if ls or il:
    P("## E3/E4. Solution quality: inter-route local search and ruin-and-recreate (2 threads)\n")
    allc = [i for i in ORDER if cp.get(i) and ls.get(i)]
    P("| Pipeline | Avg gap (%) | Geo-mean gap (%) | Median gap (%) | Total time (s) | Valid |")
    P("|---|---:|---:|---:|---:|:-:|")
    for name, d in [("MDS (paper's algorithm, C port)", cp), ("+ inter-route LS", ls), ("+ LS + 10 s ruin-and-recreate", il)]:
        ins = [i for i in allc if d.get(i)]
        if not ins:
            continue
        g = [gap(d[i]["costs"][-1], i) for i in ins]
        P(f"| {name} ({len(ins)} inst) | {st.mean(g):.2f} | {gmean(g):.2f} | {st.median(g):.2f} | {sum(d[i]['times'][-1] for i in ins):.1f} | {'yes' if all(d[i]['valid'] for i in ins) else 'NO'} |")
    P("")
    P("Per instance (gap % to BKS):\n")
    P("| Instance | BKS | MDS | +LS | +LS+ILS10s | time MDS (s) | time +LS (s) |")
    P("|---|---:|---:|---:|---:|---:|---:|")
    for i in allc:
        g0 = gap(cp[i]["costs"][-1], i)
        g1 = gap(ls[i]["costs"][-1], i)
        g2 = gap(il[i]["costs"][-1], i) if il.get(i) else float("nan")
        P(f"| {i} | {BKS[i]:g} | {g0:.2f} | {g1:.2f} | {g2:.2f} | {cp[i]['times'][-1]:.2f} | {ls[i]['times'][-1]:.2f} |")
    P("")

it = load("iters.csv")
if it:
    P("## E5. MDS iterations vs final quality (LS + 10 s ILS, 2 threads)\n")
    P("Gap (%) to BKS per instance; last columns: average gap and total wall time.\n")
    P("| rho | " + " | ".join(sub) + " | avg gap | total time (s) |")
    P("|---|" + "---:|" * (len(sub) + 2))
    for lab in it:
        gs = [gap(it[lab][i]["costs"][-1], i) if it[lab].get(i) else float("nan") for i in sub]
        tt = sum(it[lab][i]["times"][-1] for i in sub if it[lab].get(i))
        P(f"| {lab} | " + " | ".join(f"{g:.2f}" for g in gs) + f" | {st.mean([g for g in gs if g == g]):.2f} | {tt:.0f} |")
    P("")

eq = load("equal_time.csv")
if eq:
    P("## E6. Same total time budget on the largest instances: spend it on MDS iterations or on ruin-and-recreate?\n")
    big = [i for i in sub if any(eq[l].get(i) for l in eq)]
    P("| Config | " + " | ".join(big) + " | avg gap |")
    P("|---|" + "---:|" * (len(big) + 1))
    for lab in eq:
        gs = [gap(eq[lab][i]["costs"][-1], i) if eq[lab].get(i) else float("nan") for i in big]
        ts = [eq[lab][i]["times"][-1] if eq[lab].get(i) else float("nan") for i in big]
        P(f"| {lab} | " + " | ".join(f"{g:.2f} ({t:.0f}s)" for g, t in zip(gs, ts)) + f" | {st.mean([g for g in gs if g == g]):.2f} |")
    P("")

open(os.path.join(R, "summary.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

# ---------------- E7: against the paper's reported numbers ----------------
lines2 = []
Q = lines2.append
pit = load("c_par2_it2500.csv").get("c_par2_it2500", {})
lit = load("c_ls_it2500.csv").get("c_ls_it2500", {})
if pit:
    paper = {"X-n1001-k43": (7.43, 0.20), "X-n979-k58": (7.47, 0.22), "Golden_12": (3.21, 0.09),
             "Golden_16": (3.51, 0.08), "Brussels2": (173.42, 20.42), "Flanders2": (373.58, 79.37)}
    Q("## E7. Against the times reported in the paper (40-thread Xeon E5-2640 v4)\n")
    Q("The paper's ParMDS loop is `for (i = 0; i < 1e5; i += nthreads)`, so with 40 threads it runs **2,500** "
      "randomized iterations per instance, not 1e5. The rows below run the C version with the same 2,500 iterations on **2 threads**.\n")
    Q("| | Machine | Threads | Iterations | Total time, 130 instances (s) | Avg gap (%) |")
    Q("|---|---|---:|---:|---:|---:|")
    Q("| Paper: SeqMDS | Xeon E5-2640 v4 | 1 | 1e5 | 1722.44 | - |")
    Q("| Paper: ParMDS (strided) | Xeon E5-2640 v4 | 40 | 2,500 | 186.50 | 11.85 (reported) |")
    for name, d in [("This work: MDS, C, 2 threads", pit), ("This work: MDS + inter-route LS, 2 threads", lit)]:
        ins = [i for i in ORDER if d.get(i)]
        if len(ins) == 0: continue
        g = [gap(d[i]["costs"][-1], i) for i in ins]
        Q(f"| {name} ({len(ins)} inst) | Xeon @2.8 GHz | 2 | 2,500 | {sum(d[i]['times'][-1] for i in ins):.2f} | {st.mean(g):.2f} |")
    Q("")
    Q("| Instance | Paper SeqMDS (s) | Paper ParMDS 40 thr (s) | C, 2 thr, 2,500 it (s) | C + LS, 2 thr, 2,500 it (s) | gap paper-style MDS (%) | gap with LS (%) |")
    Q("|---|---:|---:|---:|---:|---:|---:|")
    for i, (s_, p_) in paper.items():
        a = pit.get(i); b = lit.get(i)
        if a:
            Q(f"| {i} | {s_} | {p_} | {a['times'][-1]:.2f} | {b['times'][-1] if b else float('nan'):.2f} | {gap(a['costs'][-1], i):.2f} | {gap(b['costs'][-1], i) if b else float('nan'):.2f} |")
    Q("")
if lines2:
    with open(os.path.join(R, "summary.md"), "a") as f:
        f.write("\n".join(lines2) + "\n")
    print("\n".join(lines2))
