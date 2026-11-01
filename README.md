# cmds: the MDS heuristic for CVRP in C, faster and with better solutions

A C re-implementation of **parMDS** (Rajesh Pandian M, Somesh Singh, Rupesh Nasre, N.S. Narayanaswamy,
*"Effective Parallelization of the Vehicle Routing Problem"*, GECCO 2023,
[paper](https://dl.acm.org/doi/10.1145/3583131.3590458), [code](https://github.com/mrprajesh/parMDS), MIT licence).
It has three parts:

1. **The same MDS algorithm** in C (MST, then randomized DFS, then greedy split, then per-route 2-opt), parallelised with
   OpenMP (`simd`, user-defined `reduction`, `schedule(runtime)` for static/dynamic/guided/auto) or pthreads.
2. **Inter-route local search** (relocate 1–3-customer segments, swap, four 2-opt* variants, intra-route 2-opt)
   on k-nearest-neighbour ("granular") neighbourhoods.
3. **Parallel ruin-and-recreate** (an iterated local search with simulated-annealing acceptance) for a fixed time budget.

See `results/summary.md` for all measurements, and `REPORT.md` for the write-up.

## Build

```bash
make            # builds ./cmds (needs gcc with OpenMP)
make test       # runs the toy instance and one benchmark instance
```

or directly: `gcc -O3 -march=native -fopenmp cmds.c -o cmds -lm -lpthread`

## Run

```bash
./cmds instance.vrp [options]
```

| Option | Meaning | Default |
|---|---|---|
| `-nthreads n` | threads | 2 |
| `-round 0/1` | round distances (use 0 for Golden and CMT, as in the paper) | 1 |
| `-iters rho` | randomized DFS iterations | 100000 |
| `-par omp/pthread` | runtime for the randomized phase | omp |
| `-sched static/dynamic/guided/auto` | loop schedule | static |
| `-chunk c` | chunk size | runtime default |
| `-ls 0/1` | 0 = exactly the paper's pipeline; 1 = add inter-route local search | 1 |
| `-starts S` | run local search on the S best distinct MDS solutions | 1 |
| `-time sec` | seconds of parallel ruin-and-recreate after local search | 0 |
| `-knn k` | neighbourhood size | 40 |
| `-temp f` | starting annealing temperature (fraction of mean edge length) | 0.5 |
| `-matrix 0/1` | precompute the distance matrix (if it fits in 1 GiB) | 0 |
| `-seed s` | random seed | 1 |
| `-quiet` | don't print routes | |
| `-sol file` | also write the solution to a file | |

Examples:

```bash
./cmds X-n101-k25.vrp -ls 0                      # the paper's algorithm only
./cmds X-n101-k25.vrp                            # + inter-route local search
./cmds Golden_12.vrp -round 0 -time 10           # + 10 s of ruin-and-recreate
./cmds Antwerp1.vrp -nthreads 2 -sched dynamic -chunk 64 -par pthread
```

**Output.** One line of statistics goes to stderr:

```
file Cost <after 1 DFS> <after rho DFS> <after post-processing> [LS <cost>] [ILS <cost>]
     Time(seconds) <t1> <t2> <t3> [<t4>] [<t5>] threads <n> routes <k> VALID
```

The solution goes to stdout in CVRPLIB format (`Route #i: ...`, then `Cost ...`, with the depot = 0).
Every run checks feasibility (each customer visited exactly once, no route over capacity) and recomputes the cost from scratch.

## What changed compared to the original C++ code

| | Original parMDS | This version |
|---|---|---|
| Language | C++ (STL) | C99 + OpenMP / pthreads |
| MST | Prim with `std::set` on an explicit n×n edge list (about 16·n² bytes, so 14 GB for Flanders2) | Array-based O(n²) Prim, computing distances on the fly (O(n) memory), `omp for simd` key update, user-defined arg-min `reduction` |
| Randomized loop | All threads shuffle one **shared** MST and write `minCost`/`minRoute` without a lock, and call non-thread-safe `rand()`. Built with `g++ -fopenmp`, it returns **invalid solutions** | Thread-private state, per-iteration seeding (results identical for any thread count or schedule), merge in a critical section at the end |
| Iteration count | `i += nthreads`, so it does only 1e5/nthreads iterations | Always rho iterations in total |
| DFS + split | Recursive DFS, then a separate split, then a separate cost pass | One fused pass, iterative DFS on a children-only rooted tree, children shuffled lazily, early abandon once the cost exceeds the best so far |
| 2-opt | Recomputes the whole route cost for every candidate: O(n³) per pass | Delta evaluation: O(n²) per pass, giving identical moves |
| Post-processing | `push_back` into a shared vector inside `omp parallel for` (a race) | Each route written in place, `schedule(dynamic,1)` |
| Inter-route moves | none | relocate / swap / 2-opt* / intra 2-opt with a work queue |
| Metaheuristic | none | parallel ruin-and-recreate (SISR-style) with simulated annealing |

## Reproducing the experiments

Clone the original repo next to this folder (`git clone https://github.com/mrprajesh/parMDS ../parMDS`), then build
the baselines and run everything:

```bash
make cmds cmds_nosimd
g++ -O3 -std=c++14 ../parMDS/seqMDS.cpp -o orig_seq
g++ -O3 -std=c++14 -fopenmp ../parMDS/parMDS.cpp -o orig_par
gcc -O3 -march=native -fopenmp variants/cmds_prim_v1.c -o /tmp/cmds_v1 -lm   # Prim v1/v2, for the SIMD table only
./bench.sh orig_seq results/orig_seq.csv all.txt ./orig_seq
./run_experiments.sh
python3 analyze.py        # writes results/summary.md
```

## Files

- `cmds.c`: the solver
- `Makefile`
- `bench.sh`: runs a command over a list of instances and appends one CSV line per instance
- `run_experiments.sh`: all experiments in the report
- `analyze.py`: builds `results/summary.md` from the CSVs
- `check_sol.py`: independent Python checker (feasibility + cost) for `.sol` files
- `bks.txt`: best-known solution values (CVRPLIB) for the 130 instances
- `results/`: raw CSVs and `summary.md` (`results/v1/` holds the runs made with the first Prim version)
- `sols/`: example solutions (10 instances, LS + 3 s ruin-and-recreate), all verified by `check_sol.py`
- `variants/`: the two earlier Prim versions used in the SIMD/data-layout experiment
- `REPORT.md`: write-up of the method, the experiments and the results
- `all.txt`, `subset.txt`: instance lists
