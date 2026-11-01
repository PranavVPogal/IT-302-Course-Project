# Parallel CVRP in C: re-implementing and extending parMDS

**Base paper:** Rajesh Pandian M, S. Singh, R. Nasre, N.S. Narayanaswamy, *Effective Parallelization of the Vehicle
Routing Problem*, GECCO 2023 ([DOI](https://dl.acm.org/doi/10.1145/3583131.3590458), [code](https://github.com/mrprajesh/parMDS)).

**Aims:**
1. Re-implement the paper's MDS algorithm in C without changing the algorithm.
2. Use OpenMP (`simd`, `reduction`, `schedule(static|dynamic|guided|auto)`) and pthreads to beat the paper's running times.
3. Add inter-route optimisation to get better CVRP solutions than parMDS.

## 1. Summary of results

All runs use the paper's 130 benchmark instances (CVRPLIB X, Golden, CMT, E, F, P, tai and the large Belgian instances of up to 30,000 customers).
Best-known solutions (BKS) come from CVRPLIB. The machine is a **2-core** Xeon @ 2.8 GHz; the paper used 40 threads.

| Result | Number |
|---|---|
| Speed-up of the C port over the original seqMDS, same algorithm, 1 thread (128 instances) | **4.33x** (1327.6 s → 306.5 s) |
| Same, 2 threads (OpenMP) | **8.09x** (164.2 s), with 1.9x parallel speed-up on 2 cores |
| Same work as the paper's 40-thread ParMDS (2,500 iterations): total time for all 130 instances | **8.2 s on 2 threads**, vs **186.5 s** reported on 40 threads |
| Memory | The original needs an n×n edge list and crashes (`bad_alloc`) on Flanders1/2 in 7 GB. The C port uses O(n) memory and solves both |
| Original parMDS built with a standard OpenMP compiler (`g++ -fopenmp`) | **INVALID solutions on every instance tested** (data races) |
| Average gap to BKS: paper's algorithm (C port) | 12.34 % |
| + inter-route local search (+1% time) | **6.08 %** |
| + LS + 10 s parallel ruin-and-recreate | **0.71 %** (median 0.52 %; 15 instances match the BKS) |

All 130 solutions in every run pass the solver's own feasibility check. A sample was also re-verified with an
independent Python checker (`check_sol.py`).

## 2. The algorithm (unchanged from the paper)

1. **MST**: Prim's algorithm on the complete graph of depot + customers.
2. **DFS**: a preorder DFS of the MST from the depot gives a giant tour (the classic MST 2-approximation idea for TSP).
3. **Split**: walk the giant tour and start a new vehicle whenever the next customer would exceed the capacity.
4. **Randomise**: shuffle every node's MST adjacency list, repeat steps 2–3 ρ = 10⁵ times, and keep the cheapest.
5. **Post-process** each route: (nearest-neighbour + 2-opt) vs (2-opt). Keep the better one.

With `-ls 0` the C program runs exactly this pipeline. Its solution quality matches the original: 12.39 % vs 12.48 % average gap
on the same 128 instances. The difference comes only from different random sequences.

## 3. Implementation in C: what changed and why it is faster

| Component | Original C++ | C version | Effect |
|---|---|---|---|
| Distances | n(n−1)/2 doubles **and** an n×n `vector<Edge>` complete graph (about 16·n² B) | Computed on the fly from coordinates (O(n) memory). An optional matrix (`-matrix 1`) gave no gain (E2) | Flanders1/2 become solvable; less memory traffic |
| Prim | `std::set` priority queue over n² edges | Array-based O(n²) Prim over a **compacted structure-of-arrays** of unvisited nodes. The key update is fused with a SIMD min-reduction | Prim on Flanders2: 6.8 s (v1) → 0.61 s (1 thread) |
| Random iteration | Shuffle all adjacency lists, recursive DFS into a vector, split into `vector<vector>`, then a separate cost pass | **One fused pass**: iterative DFS on a children-only rooted tree, children shuffled lazily when a node is reached (Fisher-Yates, xoshiro256** RNG), split and cost accumulated on the fly, abandoned once the partial cost exceeds the best | Most of the 4.3x |
| 2-opt | Rebuilds the route and re-sums its full cost for every (i,k): **O(L³)** per pass | Delta evaluation, **O(L²)** per pass, identical moves | Post-processing on long Belgian routes becomes negligible |
| Determinism | – | Iteration *i* is seeded with hash(seed, *i*) and shuffles the *original* tree order | **Same result for any thread count, schedule or runtime**. In E2 every configuration gives the same cost, so only time is compared |

## 4. Parallelisation

**What runs in parallel:**
- the ρ randomised iterations (embarrassingly parallel, ~95 % of the time)
- Prim's arg-min and key update
- the per-route post-processing
- the k-nearest-neighbour lists
- the local search on several starting solutions (`-starts`)
- the ruin-and-recreate search (one independent search per thread, synchronised periodically)

**OpenMP constructs used:**
- `parallel for schedule(runtime)` with `omp_set_schedule()`, so static/dynamic/guided/auto and the chunk size are command-line options
- `for simd` and `simd reduction(min:…)` in Prim
- `simd reduction(+:…)` for route costs
- `reduction(+:…)` for ILS iteration counts
- `schedule(dynamic,1)` for post-processing (routes vary greatly in length)
- `single`, `barrier` and `critical` for merging thread-local results

**pthreads:** the same loop with either a static block partition or dynamic chunks taken from an atomic counter (`__atomic_fetch_add`).

**Races in the original that this version avoids:**
1. All threads shuffle the *same* shared MST adjacency lists.
2. `minCost` and `minRoute` (a `vector<vector<int>>`) are updated with no lock.
3. `rand()` is not thread-safe.
4. In post-processing, `push_back` into a shared vector happens inside `omp parallel for`.

With `g++ -fopenmp` these races produce solutions that are not valid (for example "cost 3" for Golden_12); see the table in
`results/summary.md`. The paper used `nvc++ -acc=multicore`, an OpenACC flag, so its OpenMP pragmas may not have taken
effect the way they do under gcc. In the C version every thread has private scratch space and its own pool of best solutions,
and these are merged in a critical section once at the end.

## 5. Experiments

Setup: gcc 13.3 `-O3 -march=native -fopenmp`, 2-core Xeon @ 2.8 GHz with AVX-512, 7 GB RAM.
Golden and CMT instances are run with unrounded distances and all others with rounded distances, as in the paper's `runRounds.sh`.
Raw data is in `results/*.csv`; every table is generated by `analyze.py` into `results/summary.md`.

### E1. Original vs C port (same algorithm)

| Version | Total time, 128 instances (s) | Speed-up | Avg gap (%) |
|---|---:|---:|---:|
| Original seqMDS (C++) | 1327.6 | 1.00x | 12.48 |
| C port, 1 thread | 306.5 | 4.33x | 12.39 |
| C port, 2 threads | 164.2 | 8.09x | 12.39 |

Examples: X-n1001-k43 takes 7.80 s → 2.36 s (1 thread) → 1.13 s (2 threads), and Brussels2 takes 251.7 s → 48.1 s → 25.5 s.
The two Flanders instances (20k and 30k customers), which the original cannot run, take 30 s and 50 s on 2 threads.

### E2. Loop schedules and runtimes (2 threads, ρ = 10⁵, 10 instances from 100 to 30,000 customers)

| Config | Total (s) | | Config | Total (s) |
|---|---:|---|---|---:|
| static (default) | 91.7 | | guided | 88.9 |
| static, chunk 1 | 90.1 | | **guided, chunk 64** | **87.1** |
| static, chunk 64 | 92.6 | | auto | 88.6 |
| dynamic, chunk 1 | 90.5 | | pthreads, static blocks | 88.1 |
| dynamic, chunk 64 | 90.5 | | pthreads, dynamic (atomic counter) | 90.2 |
| dynamic, chunk 1024 | 91.3 | | precomputed distance matrix | 91.7 |

**Takeaway.** The spread is only about 6 %, and guided scheduling is marginally the best. Each iteration is a full DFS of the
tree (microseconds to milliseconds) and all iterations do nearly the same work. So scheduling overhead is negligible even
with chunk 1, and load imbalance is small. The only variation comes from the early-abandon cut-off. pthreads and OpenMP perform
the same. These runs used the first Prim version. The E1 times above were re-measured with the final code.

### E3. SIMD and data layout (Prim's MST, time in seconds)

| Prim version | Leuven2 (4k) | Antwerp1 (6k) | Ghent2 (11k) | Flanders2 (30k) |
|---|---:|---:|---:|---:|
| v1: original layout, `in[]` flag (pragma present but **not vectorized**), 1 thr | 0.132 | 0.238 | 0.869 | 6.80 |
| v2: compacted SoA of unvisited nodes, 1 thr | 0.025 | 0.049 | 0.141 | 1.01 |
| v3: v2 + fused SIMD min-reduction, SIMD **off**, 1 thr | 0.021 | 0.042 | 0.117 | 0.61 |
| v3, SIMD **on**, 1 thr | 0.018 | 0.035 | 0.119 | 0.62 |
| v3, SIMD **on**, 2 thr | 0.017 | 0.032 | 0.079 | 0.39 |

**Takeaway.** gcc's vectorization report showed the first `omp for simd` loop was *not* vectorized, because of mixed
`char`/`int`/`double` arrays and a data-dependent test. Fixing the data layout gave 6.7x. Fusing the arg-min into the update
pass gave another 1.65x. Switching SIMD on or off then changes ≤10 %, because the loop moves about 36 bytes per element for a
handful of flops, so it is bound by memory and cache bandwidth rather than arithmetic.
All three versions produce the identical MST on all 130 instances (verified).

### E4. Inter-route optimisation (quality)

The local search is applied to the MDS solution. Its moves are:
- **relocate** a segment of 1–3 customers, optionally reversed, next to a neighbour
- **swap** two customers in different routes
- **2-opt\***: tail exchange between two routes, in 4 variants including the reversing ones
- **intra-route 2-opt**

It is restricted to each customer's 40 nearest neighbours ("granular" neighbourhoods), uses first-improvement and a
work queue so only nodes near a change are re-examined, and checks capacity in O(1) with prefix loads.

The ruin-and-recreate search repeatedly does three things:
1. Removes a cluster of 5–30 nearby customers.
2. Re-inserts them with the cheapest insertion next to one of their neighbours, using random, demand-descending or far-first order with 1 % "blinks".
3. Runs the local search, and accepts the result with simulated annealing (start temperature 0.5 × mean edge length, cooling linearly).

Each thread searches independently and adopts the global best at 8 synchronisation points.

| Pipeline (2 threads, 130 instances) | Avg gap | Geo-mean gap | Median gap | Total time |
|---|---:|---:|---:|---:|
| MDS (paper's algorithm) | 12.34 % | 10.95 % | 11.10 % | 243.8 s |
| + inter-route LS | **6.08 %** | 5.40 % | 5.46 % | 246.5 s |
| + LS + 10 s ruin-and-recreate | **0.71 %** | 0.43 % | 0.52 % | 1564 s |

The local search costs only 2.7 s across all 130 instances but halves the gap, which is far more than the 10⁵ random
DFS iterations achieve. With 10 s of ruin-and-recreate:
- 15 instances match the BKS (all the E, F, P instances and 10 X instances)
- 105 of 130 are within 1 % of BKS
- the worst cases are the largest Belgian instances (Ghent2 5.4 %, Brussels2 6.1 %, Flanders2 6.4 %), where 10 s is very little search for 10–30k customers

### E5/E6. Where should the time go?

With LS + 10 s ILS, ρ = 1,000 gives the same final quality as ρ = 10⁵ (2.00 % vs 1.99 % on the 10-instance subset) in 107 s
instead of 194 s. Given the *same total time* on the four largest instances, spending it on ruin-and-recreate instead of
MDS iterations is clearly better:

| Same time budget | Leuven2 | Antwerp1 | Ghent2 | Flanders2 | Avg gap |
|---|---:|---:|---:|---:|---:|
| ρ = 10⁵ + 10 s ILS | 3.48 % | 1.30 % | 5.64 % | 6.48 % | 4.23 % |
| ρ = 10³ + the rest of the time in ILS | 3.34 % | 1.02 % | 4.91 % | 4.86 % | **3.53 %** |

So the randomised MDS phase is best used as a fast, good initial solution generator, not as the main search.

### E7. Against the times in the paper

The paper's ParMDS loop is `for (i = 0; i < 100000; i += nthreads)`, so its reported 40-thread runs do **2,500**
iterations per instance. Running the C version with the same 2,500 iterations on just 2 threads:

| | Machine | Threads | Total time, 130 instances | Avg gap |
|---|---|---:|---:|---:|
| Paper: SeqMDS | Xeon E5-2640 v4 | 1 | 1722.4 s | – |
| Paper: ParMDS | Xeon E5-2640 v4 | 40 | 186.5 s | 11.85 % (as reported) |
| **C version, MDS** | Xeon @ 2.8 GHz | 2 | **8.2 s** | 13.28 % |
| **C version, MDS + inter-route LS** | Xeon @ 2.8 GHz | 2 | **13.3 s** | **6.12 %** |

Per instance: Flanders2 takes 79.4 s (paper, 40 threads) vs 1.8 s here (3.5 s with LS), and Brussels2 takes 20.4 s vs 0.9 s.
The hardware differs: our cores are somewhat faster, but we use 2 threads instead of 40. Most of the gap comes from the O(L³)
post-processing and the memory-heavy MST in the original, not from thread count. Our "MDS" gap of 13.28 % uses our own BKS
table and averaging, so it is not directly comparable with the paper's 11.85 % figure.

## 6. Limitations and honest caveats

- Only 2 cores were available, so the speed-up from more threads is not measured here. The randomised phase is
  embarrassingly parallel and should keep scaling; the paper observed flattening beyond 8 threads on Flanders2.
- The ruin-and-recreate results are time-bounded and therefore not bit-for-bit reproducible. The MDS phase and the local search are.
- BKS values for X and Belgian instances come from CVRPLIB solution files. Golden, CMT, E, F, P and tai385 values are from
  the literature, and tai385 with rounded distances is taken as 24366.
- E2 was measured with the first Prim version. Its effect on times is under 3.5 s, and only on Flanders2.

## 7. Reproducing

```bash
make
./run_experiments.sh      # all experiments (≈ 1.5 h on 2 cores); expects the original repo next to this one
python3 analyze.py        # → results/summary.md
python3 check_sol.py X-n101-k25.vrp sol.sol 1   # independent solution checker
```
