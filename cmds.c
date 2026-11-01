/*
 * cmds.c  --  MDS heuristic for the Capacitated Vehicle Routing Problem, in C.
 *
 * Re-implementation (in C, OpenMP + pthreads) of the MDS algorithm from:
 *   Rajesh Pandian M, Somesh Singh, Rupesh Nasre, N.S. Narayanaswamy,
 *   "Effective Parallelization of the Vehicle Routing Problem", GECCO 2023.
 *   Original C++ code: https://github.com/mrprajesh/parMDS (MIT licence)
 *
 * Algorithm (unchanged from the paper):
 *   1. Prim's MST over the complete graph (depot + customers).
 *   2. DFS of the MST from the depot  -> one giant tour.
 *   3. Greedy split of the giant tour into capacity-feasible routes.
 *   4. Repeat 2-3 with randomly shuffled MST adjacency lists (rho = 1e5 times),
 *      keep the cheapest.
 *   5. Per-route post-processing: (nearest-neighbour + 2-opt) vs (2-opt),
 *      keep the better one.
 *
 * Additions in this version (all optional, see -ls / -starts / -time):
 *   6. Inter-route local search on granular (k-nearest) neighbourhoods:
 *      relocate of 1..3-customer segments (with reversal), swap,
 *      2-opt* (4 variants) and intra-route 2-opt.
 *   7. Multi-start: local search on the best S distinct MDS solutions in parallel.
 *   8. Time-bounded parallel ruin-and-recreate (iterated local search).
 *
 * Build : gcc -O3 -march=native -fopenmp cmds.c -o cmds -lm -lpthread
 * Usage : ./cmds file.vrp [options]   (run without args for the option list)
 *
 * stderr : one-line stats; stdout : solution in CVRPLIB format (depot = 0).
 */
#define _GNU_SOURCE
#include <float.h>
#include <limits.h>
#include <math.h>
#include <omp.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------------ */
/* Instance data (global, read-only after loading)                          */
/* ------------------------------------------------------------------------ */
static int N;             /* number of nodes incl. depot (node 0)           */
static double CAP;        /* vehicle capacity                               */
static float *X, *Y;      /* coordinates (float, as the original's stof)    */
static double *DEM;       /* demands                                        */
static int ROUND = 1;     /* round distances to nearest integer             */
static double *DM = NULL; /* optional full distance matrix                  */

#define EPS 1e-9

static inline double dist(int i, int j) {
  if (DM) return DM[(size_t)i * N + j];
  double dx = (double)X[i] - (double)X[j];
  double dy = (double)Y[i] - (double)Y[j];
  double w = sqrt(dx * dx + dy * dy);
  return ROUND ? round(w) : w;
}

static void *xmalloc(size_t n) {
  void *p = malloc(n ? n : 1);
  if (!p) { fprintf(stderr, "out of memory\n"); exit(1); }
  return p;
}
static void *xcalloc(size_t n, size_t s) {
  void *p = calloc(n ? n : 1, s);
  if (!p) { fprintf(stderr, "out of memory\n"); exit(1); }
  return p;
}

/* ------------------------------------------------------------------------ */
/* Parameters                                                               */
/* ------------------------------------------------------------------------ */
typedef enum { PAR_OMP, PAR_PTHREAD } par_t;
static struct {
  int nthreads;
  long iters;        /* rho: number of randomized DFS iterations          */
  par_t par;         /* omp | pthread                                     */
  omp_sched_t sched; /* static | dynamic | guided | auto                  */
  int chunk;         /* chunk size (0 = runtime default)                  */
  int ls;            /* 0 = original post-processing only, 1 = + inter-route LS */
  int starts;        /* number of MDS solutions to improve (multi-start)  */
  double time;       /* seconds of ruin-and-recreate after LS (0 = off)   */
  int knn;           /* granular neighbourhood size                       */
  int matrix;        /* precompute distance matrix (if it fits)           */
  uint64_t seed;
  double temp;       /* initial SA temperature, as a fraction of the mean edge */
  int quiet;         /* don't print routes                                */
  const char *solfile;
} P = {2, 100000, PAR_OMP, omp_sched_static, 0, 1, 1, 0.0, 40, 0, 1, 0.50, 0, NULL};

/* ------------------------------------------------------------------------ */
/* RNG: xoshiro256** seeded by splitmix64 (fast, thread-local)              */
/* ------------------------------------------------------------------------ */
typedef struct { uint64_t s[4]; } rng_t;
static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
static inline uint64_t splitmix(uint64_t *x) {
  uint64_t z = (*x += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}
static inline void rseed(rng_t *r, uint64_t seed) {
  for (int i = 0; i < 4; i++) r->s[i] = splitmix(&seed);
}
static inline uint64_t rnext(rng_t *r) {
  uint64_t *s = r->s, res = rotl(s[1] * 5, 7) * 9, t = s[1] << 17;
  s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t;
  s[3] = rotl(s[3], 45);
  return res;
}
static inline uint32_t rbound(rng_t *r, uint32_t n) { /* uniform in [0,n) */
  return (uint32_t)(((rnext(r) >> 32) * (uint64_t)n) >> 32);
}
static inline double runif(rng_t *r) { return (rnext(r) >> 11) * 0x1.0p-53; }

/* ------------------------------------------------------------------------ */
/* Input                                                                    */
/* ------------------------------------------------------------------------ */
static void read_vrp(const char *fn) {
  FILE *f = fopen(fn, "r");
  if (!f) { fprintf(stderr, "Could not open the file \"%s\"\n", fn); exit(1); }
  char line[4096];
  N = 0; CAP = 0;
  while (fgets(line, sizeof line, f)) {
    char *c = strchr(line, ':');
    if (strstr(line, "DIMENSION") && c) N = atoi(c + 1);
    else if (strstr(line, "CAPACITY") && c) CAP = strtof(c + 1, NULL);
    else if (strstr(line, "NODE_COORD_SECTION")) {
      X = xmalloc(sizeof(float) * N); Y = xmalloc(sizeof(float) * N);
      for (int i = 0; i < N; i++)
        if (fscanf(f, "%*d %f %f", &X[i], &Y[i]) != 2) { fprintf(stderr, "bad coords\n"); exit(1); }
    } else if (strstr(line, "DEMAND_SECTION")) {
      DEM = xmalloc(sizeof(double) * N);
      for (int i = 0; i < N; i++) {
        float d;
        if (fscanf(f, "%*d %f", &d) != 1) { fprintf(stderr, "bad demand\n"); exit(1); }
        DEM[i] = d;
      }
    } else if (strstr(line, "DEPOT_SECTION") || strncmp(line, "EOF", 3) == 0) break;
  }
  fclose(f);
  if (N < 2 || CAP <= 0 || !X || !DEM) { fprintf(stderr, "invalid instance\n"); exit(1); }
}

static void build_matrix(void) {
  size_t bytes = (size_t)N * N * sizeof(double);
  if (bytes > (size_t)1 << 30) return; /* > 1 GiB: compute on the fly */
  double *M = xmalloc(bytes);
#pragma omp parallel for schedule(dynamic, 16) num_threads(P.nthreads)
  for (int i = 0; i < N; i++) {
#pragma omp simd
    for (int j = 0; j < N; j++) {
      double dx = (double)X[i] - (double)X[j], dy = (double)Y[i] - (double)Y[j];
      double w = sqrt(dx * dx + dy * dy);
      M[(size_t)i * N + j] = ROUND ? round(w) : w;
    }
  }
  DM = M;
}

/* ------------------------------------------------------------------------ */
/* Step 1: Prim's MST, O(n^2), without building the complete graph.         */
/* Unvisited nodes are kept compacted in structure-of-arrays form, so every  */
/* loop is a branch-free, unit-stride "omp for simd" loop that shrinks each  */
/* step. The key update is fused with a SIMD min-reduction of the new keys;  */
/* a second SIMD min-reduction picks the smallest node id among the keys     */
/* equal to that minimum (the same tie-break as a scalar arg-min). Squared   */
/* distances give the same MST as distances.                                 */
/* ------------------------------------------------------------------------ */
static int *prim_mst(void) {
  int *par = xmalloc(sizeof(int) * N);
  int m = N - 1;                                   /* nodes not yet in the tree */
  double *rx = xmalloc(sizeof(double) * N), *ry = xmalloc(sizeof(double) * N);
  double *rk = xmalloc(sizeof(double) * N);
  int *rid = xmalloc(sizeof(int) * N), *rp = xmalloc(sizeof(int) * N);
  int *where = xmalloc(sizeof(int) * N);           /* node id -> position in the arrays */
  par[0] = -1;
  double mk = DBL_MAX;                             /* smallest key among unvisited */
  for (int p = 0; p < m; p++) {                    /* the depot (0) starts the tree */
    int v = p + 1;
    rid[p] = v; where[v] = p; rx[p] = X[v]; ry[p] = Y[v]; rp[p] = 0;
    double dx = rx[p] - (double)X[0], dy = ry[p] - (double)Y[0];
    rk[p] = dx * dx + dy * dy;
    if (rk[p] < mk) mk = rk[p];
  }
  int nt = (N >= 3000) ? P.nthreads : 1;           /* fork/barrier cost not worth it below */
  int bid;
  double ux = 0, uy = 0;
  int u = 0;
#pragma omp parallel num_threads(nt)
  while (m > 0) {
    /* 1) among keys equal to the minimum, pick the smallest node id
     *    (SIMD min-reduction; same tie-break as a scalar arg-min)    */
#pragma omp single
    bid = INT_MAX;
    const double mkk = mk;
#pragma omp for simd reduction(min : bid) schedule(static)
    for (int p = 0; p < m; ++p) {
      int c = rk[p] == mkk ? rid[p] : INT_MAX;
      bid = c < bid ? c : bid;
    }
#pragma omp single
    { /* add it to the tree, remove it from the arrays */
      int q = where[bid];
      u = rid[q]; par[u] = rp[q]; ux = rx[q]; uy = ry[q];
      m--;
      rx[q] = rx[m]; ry[q] = ry[m]; rk[q] = rk[m]; rid[q] = rid[m]; rp[q] = rp[m];
      where[rid[q]] = q;
      mk = DBL_MAX;
    }
    /* 2) fused key update + new minimum (SIMD min-reduction) */
    const int mm = m, uu = u;
    const double uxx = ux, uyy = uy;
#pragma omp for simd reduction(min : mk) schedule(static)
    for (int p = 0; p < mm; ++p) {
      double dx = rx[p] - uxx, dy = ry[p] - uyy;
      double d = dx * dx + dy * dy;
      int b = d < rk[p];
      double nk = b ? d : rk[p];
      rk[p] = nk;
      rp[p] = b ? uu : rp[p];
      mk = nk < mk ? nk : mk;
    }
  }
  free(rx); free(ry); free(rk); free(rid); free(rp); free(where);
  return par;
}

/* MST rooted at the depot as a children-only CSR (off[n+1], adj[n-1]).
 * par[] from Prim is already a rooted tree (par[0] = -1). */
static void mst_csr(const int *par, int **off_out, int **adj_out) {
  int *off = xcalloc(N + 1, sizeof(int));
  int *adj = xmalloc(sizeof(int) * (N > 1 ? N - 1 : 1));
  for (int v = 1; v < N; v++) off[par[v] + 1]++;
  for (int v = 0; v < N; v++) off[v + 1] += off[v];
  int *pos = xmalloc(sizeof(int) * N);
  memcpy(pos, off, sizeof(int) * N);
  for (int v = 1; v < N; v++) adj[pos[par[v]]++] = v;
  free(pos);
  *off_out = off; *adj_out = adj;
}

/* ------------------------------------------------------------------------ */
/* Steps 2-4: one randomized iteration = shuffle + DFS + greedy split.      */
/* DFS and split are fused, and the iteration is abandoned as soon as the   */
/* partial cost exceeds the thread's current best (it can only grow).       */
/* Each iteration re-seeds from (seed, it) and shuffles the *original* MST  */
/* order, so the result is identical for any thread count / schedule.       */
/* ------------------------------------------------------------------------ */
typedef struct {
  int *buf;                 /* LIFO area for shuffled child lists, size 2n   */
  int *snode, *scur, *send; /* DFS stack: node, next child, end of its list  */
  int *order;               /* giant tour of the current iteration           */
} work_t;

static const int *G_OFF, *G_ADJ;
/* nodes packed as {x, y, demand}: one cache line per random access */
typedef struct { double x, y, d, pad; } node_t;
static node_t *ND;

static inline double distn(const node_t *a, const node_t *b) {
  double dx = a->x - b->x, dy = a->y - b->y;
  double w = sqrt(dx * dx + dy * dy);
  return ROUND ? round(w) : w;
}

static void work_alloc(work_t *w) {
  w->buf = xmalloc(sizeof(int) * 2 * N);
  w->snode = xmalloc(sizeof(int) * N);
  w->scur = xmalloc(sizeof(int) * N);
  w->send = xmalloc(sizeof(int) * N);
  w->order = xmalloc(sizeof(int) * N);
}
static void work_free(work_t *w) { free(w->buf); free(w->snode); free(w->scur); free(w->send); free(w->order); }

/* push the children of v (all MST neighbours except its parent) onto the
 * LIFO area in a uniformly random order (Fisher-Yates); returns new top */
static inline int push_children(int v, int parent, int top, int *buf, rng_t *r, int shuffle) {
  const int *off = G_OFF, *adj = G_ADJ; /* rooted CSR: children only */
  (void)parent;
  int beg = top;
  for (int e = off[v]; e < off[v + 1]; e++) buf[top++] = adj[e];
  if (shuffle)
    for (int i = top - beg - 1; i > 0; --i) {
      int j = (int)rbound(r, (uint32_t)i + 1);
      int t = buf[beg + i]; buf[beg + i] = buf[beg + j]; buf[beg + j] = t;
    }
  return top;
}

static double mds_iter(long it, work_t *w, double bound) {
  int *buf = w->buf, *snode = w->snode, *scur = w->scur, *send = w->send, *order = w->order;
  rng_t r;
  rseed(&r, P.seed * 0x9E3779B97F4A7C15ULL + (uint64_t)it);
  const int shuffle = it > 0; /* iteration 0 = the un-shuffled MST order */
  double cost = 0.0, res = CAP;
  const node_t *prev = &ND[0], *depot = &ND[0];
  int cnt = 0, sp = 0;
  int top = push_children(0, -1, 0, buf, &r, shuffle);
  snode[0] = 0; scur[0] = 0; send[0] = top; sp = 1;
  while (sp) {
    int k = sp - 1;
    if (scur[k] < send[k]) {
      int v = buf[scur[k]++];
      const node_t *nv = &ND[v];
      /* greedy split (same rule as the original convertToVrpRoutes) */
      if (res - nv->d >= 0) { cost += distn(prev, nv); res -= nv->d; }
      else { cost += distn(prev, depot) + distn(depot, nv); res = CAP - nv->d; }
      prev = nv; order[cnt++] = v;
      if (cost > bound) return DBL_MAX; /* early abandon: cost only grows */
      int beg = top;
      top = push_children(v, snode[k], top, buf, &r, shuffle);
      if (top > beg) { snode[sp] = v; scur[sp] = beg; send[sp] = top; sp++; }
    } else { sp--; top = sp ? send[sp - 1] : 0; } /* child lists start where the parent's ends */
  }
  return cost + distn(prev, depot);
}

/* Top-S pool of distinct solutions per thread (S = P.starts, usually 1). */
typedef struct {
  int S, cnt;
  double *cost; long *it; int **tour;
} pool_t;

static void pool_init(pool_t *p, int S) {
  p->S = S; p->cnt = 0;
  p->cost = xmalloc(sizeof(double) * S); p->it = xmalloc(sizeof(long) * S);
  p->tour = xmalloc(sizeof(int *) * S);
  for (int i = 0; i < S; i++) p->tour[i] = xmalloc(sizeof(int) * N);
}
static void pool_free(pool_t *p) {
  for (int i = 0; i < p->S; i++) free(p->tour[i]);
  free(p->cost); free(p->it); free(p->tour);
}
static inline int better(double c1, long i1, double c2, long i2) { return c1 < c2 || (c1 == c2 && i1 < i2); }
static inline double pool_bound(const pool_t *p) { return p->cnt < p->S ? DBL_MAX : p->cost[p->cnt - 1]; }
/* insert (sorted ascending); identical costs are treated as duplicates when S > 1 */
static void pool_add(pool_t *p, double c, long it, const int *tour) {
  if (p->cnt == p->S && !better(c, it, p->cost[p->cnt - 1], p->it[p->cnt - 1])) return;
  if (p->S > 1)
    for (int i = 0; i < p->cnt; i++) if (fabs(p->cost[i] - c) < EPS) return;
  int pos = p->cnt < p->S ? p->cnt++ : p->cnt - 1;
  int *buf = p->tour[pos];
  while (pos > 0 && better(c, it, p->cost[pos - 1], p->it[pos - 1])) {
    p->cost[pos] = p->cost[pos - 1]; p->it[pos] = p->it[pos - 1]; p->tour[pos] = p->tour[pos - 1];
    pos--;
  }
  p->cost[pos] = c; p->it[pos] = it; p->tour[pos] = buf;
  memcpy(buf, tour, sizeof(int) * (N - 1));
}
static void pool_merge(pool_t *dst, const pool_t *src) {
  for (int i = 0; i < src->cnt; i++) pool_add(dst, src->cost[i], src->it[i], src->tour[i]);
}

/* --- OpenMP driver: schedule chosen at run time (schedule(runtime)) --- */
static void random_phase_omp(pool_t *gpool) {
  omp_set_schedule(P.sched, P.chunk);
#pragma omp parallel num_threads(P.nthreads)
  {
    work_t w; work_alloc(&w);
    pool_t lp; pool_init(&lp, gpool->S);
    pool_merge(&lp, gpool); /* start with iteration 0's solution as bound */
#pragma omp for schedule(runtime) nowait
    for (long it = 1; it < P.iters; ++it) {
      double c = mds_iter(it, &w, pool_bound(&lp));
      if (c < DBL_MAX) pool_add(&lp, c, it, w.order);
    }
#pragma omp critical
    pool_merge(gpool, &lp);
    pool_free(&lp); work_free(&w);
  }
}

/* --- pthreads driver: static blocks, or dynamic chunks via an atomic counter --- */
typedef struct { int tid; pool_t *g; pthread_mutex_t *mtx; long *next; } targ_t;
static void *pt_worker(void *arg) {
  targ_t *a = arg;
  work_t w; work_alloc(&w);
  pool_t lp; pool_init(&lp, a->g->S);
  pool_merge(&lp, a->g);
  long R = P.iters - 1; /* iterations 1..iters-1 */
  if (P.sched == omp_sched_dynamic || P.sched == omp_sched_guided) {
    long chunk = P.chunk > 0 ? P.chunk : 64, s;
    while ((s = __atomic_fetch_add(a->next, chunk, __ATOMIC_RELAXED)) < R)
      for (long k = s; k < s + chunk && k < R; ++k) {
        double c = mds_iter(k + 1, &w, pool_bound(&lp));
        if (c < DBL_MAX) pool_add(&lp, c, k + 1, w.order);
      }
  } else { /* static block partition */
    long lo = R * a->tid / P.nthreads, hi = R * (a->tid + 1) / P.nthreads;
    for (long k = lo; k < hi; ++k) {
      double c = mds_iter(k + 1, &w, pool_bound(&lp));
      if (c < DBL_MAX) pool_add(&lp, c, k + 1, w.order);
    }
  }
  pthread_mutex_lock(a->mtx);
  pool_merge(a->g, &lp);
  pthread_mutex_unlock(a->mtx);
  pool_free(&lp); work_free(&w);
  return NULL;
}
static void random_phase_pthread(pool_t *gpool) {
  pthread_t *th = xmalloc(sizeof(pthread_t) * P.nthreads);
  targ_t *ta = xmalloc(sizeof(targ_t) * P.nthreads);
  pthread_mutex_t mtx = PTHREAD_MUTEX_INITIALIZER;
  pool_t snapshot; pool_init(&snapshot, gpool->S); pool_merge(&snapshot, gpool);
  long next = 0;
  for (int t = 0; t < P.nthreads; t++) {
    ta[t] = (targ_t){t, &snapshot, &mtx, &next};
    pthread_create(&th[t], NULL, pt_worker, &ta[t]);
  }
  for (int t = 0; t < P.nthreads; t++) pthread_join(th[t], NULL);
  pool_merge(gpool, &snapshot);
  pool_free(&snapshot); free(th); free(ta);
}

/* ------------------------------------------------------------------------ */
/* Solution as a list of routes                                             */
/* ------------------------------------------------------------------------ */
typedef struct {
  int nr;                 /* number of route slots in use (some may be empty) */
  int **a; int *len, *cap;
  double *load;
  int *rt, *ps;           /* route id and position of every customer          */
  double *pre;            /* cumulative load up to and including the customer */
} sol_t;

static void sol_init(sol_t *s) {
  s->nr = 0;
  s->a = xcalloc(N, sizeof(int *)); s->len = xcalloc(N, sizeof(int));
  s->cap = xcalloc(N, sizeof(int)); s->load = xcalloc(N, sizeof(double));
  s->rt = xmalloc(sizeof(int) * N); s->ps = xmalloc(sizeof(int) * N);
  s->pre = xmalloc(sizeof(double) * N);
}
static void sol_free(sol_t *s) {
  for (int r = 0; r < N; r++) free(s->a[r]);
  free(s->a); free(s->len); free(s->cap); free(s->load); free(s->rt); free(s->ps); free(s->pre);
}
static void route_fix(sol_t *s, int r) {
  double l = 0;
  int *a = s->a[r];
  for (int i = 0; i < s->len[r]; i++) {
    int v = a[i];
    s->rt[v] = r; s->ps[v] = i; l += DEM[v]; s->pre[v] = l;
  }
  s->load[r] = l;
}
static void route_set(sol_t *s, int r, const int *buf, int n) {
  if (s->cap[r] < n) {
    int c = n + n / 2 + 8;
    s->a[r] = realloc(s->a[r], sizeof(int) * c);
    if (!s->a[r]) { fprintf(stderr, "oom\n"); exit(1); }
    s->cap[r] = c;
  }
  if (buf != s->a[r]) memmove(s->a[r], buf, sizeof(int) * n);
  s->len[r] = n;
  route_fix(s, r);
}
static void sol_copy(sol_t *d, const sol_t *s) {
  for (int r = 0; r < s->nr; r++) route_set(d, r, s->a[r], s->len[r]);
  for (int r = s->nr; r < d->nr; r++) d->len[r] = 0;
  d->nr = s->nr;
  memcpy(d->rt, s->rt, sizeof(int) * N); memcpy(d->ps, s->ps, sizeof(int) * N);
  memcpy(d->pre, s->pre, sizeof(double) * N);
}
/* remove empty route slots */
static void sol_compact(sol_t *s) {
  int w = 0;
  for (int r = 0; r < s->nr; r++) {
    if (s->len[r] == 0) continue;
    if (w != r) {
      int *ta = s->a[w]; int tc = s->cap[w];
      s->a[w] = s->a[r]; s->cap[w] = s->cap[r]; s->len[w] = s->len[r];
      s->a[r] = ta; s->cap[r] = tc; s->len[r] = 0;
      route_fix(s, w);
    }
    w++;
  }
  s->nr = w;
}
static int sol_new_route(sol_t *s) {
  for (int r = 0; r < s->nr; r++) if (s->len[r] == 0) return r;
  s->len[s->nr] = 0;
  return s->nr++;
}

/* route cost: SIMD reduction over the edges of the route */
static double route_cost(const int *a, int n) {
  if (n == 0) return 0.0;
  double c = dist(0, a[0]) + dist(a[n - 1], 0);
#pragma omp simd reduction(+ : c)
  for (int j = 1; j < n; j++) c += dist(a[j - 1], a[j]);
  return c;
}
static double sol_cost(const sol_t *s) {
  double c = 0.0;
  for (int r = 0; r < s->nr; r++) c += route_cost(s->a[r], s->len[r]);
  return c;
}

/* greedy split of a giant tour into a solution */
static void sol_from_tour(sol_t *s, const int *tour) {
  int *buf = xmalloc(sizeof(int) * N);
  double res = CAP;
  int n = 0;
  s->nr = 0;
  for (int k = 0; k < N - 1; k++) {
    int v = tour[k];
    if (res - DEM[v] >= 0) { buf[n++] = v; res -= DEM[v]; }
    else { route_set(s, s->nr++, buf, n); n = 0; buf[n++] = v; res = CAP - DEM[v]; }
  }
  if (n) route_set(s, s->nr++, buf, n);
  free(buf);
}

/* ------------------------------------------------------------------------ */
/* Step 5: post-processing, same as the original postProcessIt but with an  */
/* O(n^2)-per-pass 2-opt (delta evaluation instead of re-summing the route). */
/* ------------------------------------------------------------------------ */
/* the original "tsp_approx" nearest-neighbour ordering (depot is cities[nc-1]) */
static void tsp_approx(const int *cities, int *tour, int nc) {
  int i, j, close_pt = 0;
  for (i = 1; i < nc; i++) tour[i] = cities[i - 1];
  tour[0] = cities[nc - 1];
  for (i = 1; i < nc; i++) {
    double tx = X[tour[i - 1]], ty = Y[tour[i - 1]], cd = DBL_MAX;
    for (j = nc - 1;; j--) {
      double dd = ((double)X[tour[j]] - tx) * ((double)X[tour[j]] - tx);
      if (dd <= cd) {
        dd += ((double)Y[tour[j]] - ty) * ((double)Y[tour[j]] - ty);
        if (dd <= cd) {
          if (j < i) break;
          cd = dd; close_pt = j;
        }
      }
    }
    int t = tour[i]; tour[i] = tour[close_pt]; tour[close_pt] = t;
  }
}

/* first-improvement 2-opt on a route (depot at both ends); repeat until no gain */
static void two_opt(int *c, int n) {
  if (n <= 2) return;
  int improved = 1;
  while (improved) {
    improved = 0;
    for (int i = 0; i < n - 1; i++) {
      int a = i ? c[i - 1] : 0;
      for (int k = i + 1; k < n; k++) {
        int ci = c[i], ck = c[k], b = (k < n - 1) ? c[k + 1] : 0;
        double delta = dist(a, ck) + dist(ci, b) - dist(a, ci) - dist(ck, b);
        if (delta < -EPS) {
          for (int x = i, y = k; x < y; x++, y--) { int t = c[x]; c[x] = c[y]; c[y] = t; }
          improved = 1;
        }
      }
    }
  }
}

static void postprocess(sol_t *s, int nthreads) {
#pragma omp parallel for schedule(dynamic, 1) num_threads(nthreads)
  for (int r = 0; r < s->nr; r++) {
    int n = s->len[r];
    if (n == 0) continue;
    int *cities = xmalloc(sizeof(int) * (n + 1)), *tour = xmalloc(sizeof(int) * (n + 1));
    int *r3 = xmalloc(sizeof(int) * n);
    memcpy(cities, s->a[r], sizeof(int) * n);
    cities[n] = 0;
    tsp_approx(cities, tour, n + 1);   /* tour[0] is the depot */
    int *r2 = tour + 1;
    two_opt(r2, n);                    /* route 2: NN + 2-opt */
    memcpy(r3, s->a[r], sizeof(int) * n);
    two_opt(r3, n);                    /* route 3: 2-opt      */
    if (route_cost(r3, n) > route_cost(r2, n)) memcpy(s->a[r], r2, sizeof(int) * n);
    else memcpy(s->a[r], r3, sizeof(int) * n);
    route_fix(s, r);
    free(cities); free(tour); free(r3);
  }
}

/* ------------------------------------------------------------------------ */
/* Step 6: inter-route local search on granular neighbourhoods              */
/* ------------------------------------------------------------------------ */
static int *KNN;  /* KNN[i*K + k]: k-th nearest customer of customer i */
static int K;

static void build_knn(void) {
  K = P.knn < N - 2 ? P.knn : N - 2;
  if (K < 1) K = 1;
  KNN = xmalloc(sizeof(int) * (size_t)N * K);
#pragma omp parallel num_threads(P.nthreads)
  {
    double *bd = xmalloc(sizeof(double) * (K + 1));
    int *bi = xmalloc(sizeof(int) * (K + 1));
#pragma omp for schedule(dynamic, 32)
    for (int i = 1; i < N; i++) {
      int cnt = 0;
      double xi = X[i], yi = Y[i];
      for (int j = 1; j < N; j++) {
        if (j == i) continue;
        double dx = X[j] - xi, dy = Y[j] - yi, d = dx * dx + dy * dy;
        if (cnt == K && d >= bd[K - 1]) continue;
        int p = cnt < K ? cnt++ : K - 1;
        while (p > 0 && bd[p - 1] > d) { bd[p] = bd[p - 1]; bi[p] = bi[p - 1]; p--; }
        bd[p] = d; bi[p] = j;
      }
      for (int k = 0; k < K; k++) KNN[(size_t)i * K + k] = k < cnt ? bi[k] : bi[0];
    }
    free(bd); free(bi);
  }
}

typedef struct {
  int *buf, *buf2;       /* scratch of size N */
  int *queue; unsigned char *inq; int qh, qt; /* circular work queue */
  unsigned char *mark;
  rng_t rng;
} ls_t;

static void ls_init(ls_t *c, uint64_t seed) {
  c->buf = xmalloc(sizeof(int) * (N + 1)); c->buf2 = xmalloc(sizeof(int) * (N + 1));
  c->queue = xmalloc(sizeof(int) * (N + 1)); c->inq = xcalloc(N, 1); c->mark = xcalloc(N, 1);
  c->qh = c->qt = 0;
  rseed(&c->rng, seed);
}
static void ls_free(ls_t *c) { free(c->buf); free(c->buf2); free(c->queue); free(c->inq); free(c->mark); }
static inline void push(ls_t *c, int v) {
  if (v <= 0 || c->inq[v]) return;
  c->inq[v] = 1; c->queue[c->qt] = v; c->qt = (c->qt + 1) % (N + 1);
}
static inline int pop(ls_t *c) {
  if (c->qh == c->qt) return -1;
  int v = c->queue[c->qh]; c->qh = (c->qh + 1) % (N + 1); c->inq[v] = 0;
  return v;
}

#define PRED(s, u) ((s)->ps[u] ? (s)->a[(s)->rt[u]][(s)->ps[u] - 1] : 0)
#define SUCC(s, u) ((s)->ps[u] + 1 < (s)->len[(s)->rt[u]] ? (s)->a[(s)->rt[u]][(s)->ps[u] + 1] : 0)
#define AT(s, r, i) (((i) >= 0 && (i) < (s)->len[r]) ? (s)->a[r][i] : 0)

/* ---- move application helpers ---- */
static void rev(int *a, int i, int j) { for (; i < j; i++, j--) { int t = a[i]; a[i] = a[j]; a[j] = t; } }

/* relocate segment [iu, iu+L) of route ru to after position jw of route rv */
static void apply_relocate(sol_t *s, ls_t *c, int ru, int iu, int L, int rv, int jw, int reversed) {
  int *seg = c->buf2;
  memcpy(seg, s->a[ru] + iu, sizeof(int) * L);
  if (reversed) rev(seg, 0, L - 1);
  if (ru != rv) {
    int n = s->len[ru];
    memmove(s->a[ru] + iu, s->a[ru] + iu + L, sizeof(int) * (n - iu - L));
    s->len[ru] = n - L; route_fix(s, ru);
    int m = s->len[rv], *b = c->buf;
    memcpy(b, s->a[rv], sizeof(int) * (jw + 1));
    memcpy(b + jw + 1, seg, sizeof(int) * L);
    memcpy(b + jw + 1 + L, s->a[rv] + jw + 1, sizeof(int) * (m - jw - 1));
    route_set(s, rv, b, m + L);
  } else {
    int n = s->len[ru], *b = c->buf, m = 0;
    for (int i = 0; i < n; i++) if (i < iu || i >= iu + L) b[m++] = s->a[ru][i];
    int ins = (jw < iu) ? jw + 1 : jw + 1 - L;
    int *out = s->a[ru], k = 0;
    for (int i = 0; i < ins; i++) out[k++] = b[i];
    for (int i = 0; i < L; i++) out[k++] = seg[i];
    for (int i = ins; i < m; i++) out[k++] = b[i];
    route_fix(s, ru);
  }
}

/* replace routes r1, r2 by concatenations built in buf / buf2 */
static void set_two(sol_t *s, int r1, const int *b1, int n1, int r2, const int *b2, int n2) {
  route_set(s, r1, b1, n1);
  route_set(s, r2, b2, n2);
}

/* try all moves for customer u; apply the first improving one */
static int ls_try(sol_t *s, ls_t *c, int u) {
  const int ru = s->rt[u], iu = s->ps[u], nU = s->len[ru];
  const int pu = PRED(s, u), nu = SUCC(s, u);
  const double LU = s->load[ru], preU = s->pre[u], preUex = preU - DEM[u];
  const int *nb = KNN + (size_t)u * K;

  for (int k = 0; k < K; k++) {
    const int v = nb[k];
    const int rv = s->rt[v], iv = s->ps[v], nV = s->len[rv];
    const int pv = PRED(s, v), nv = SUCC(s, v);
    const double LV = s->load[rv], preV = s->pre[v], preVex = preV - DEM[v];

    /* ---- (1) relocate segment starting at u (length 1..3) next to v ---- */
    for (int L = 1; L <= 3 && iu + L <= nU; L++) {
      const int se = s->a[ru][iu + L - 1];
      const int b = AT(s, ru, iu + L);
      const double segload = s->pre[se] - preUex;
      if (ru != rv && LV + segload > CAP + EPS) break;
      const double gain_rm = dist(pu, b) - dist(pu, u) - dist(se, b);
      for (int t = 0; t < 2; t++) {
        int jw = t == 0 ? iv : iv - 1; /* after v / before v */
        if (ru == rv && jw >= iu - 1 && jw <= iu + L - 1) continue;
        int w = AT(s, rv, jw), x = AT(s, rv, jw + 1);
        double base = gain_rm - dist(w, x);
        double d1 = base + dist(w, u) + dist(se, x);
        double d2 = L > 1 ? base + dist(w, se) + dist(u, x) : DBL_MAX;
        if (d1 < -EPS || d2 < -EPS) {
          apply_relocate(s, c, ru, iu, L, rv, jw, d2 < d1);
          push(c, u); push(c, se); push(c, pu); push(c, b); push(c, w); push(c, x); push(c, v);
          return 1;
        }
      }
    }
    if (ru != rv) {
      /* ---- (2) swap u <-> v ---- */
      if (LU - DEM[u] + DEM[v] <= CAP + EPS && LV - DEM[v] + DEM[u] <= CAP + EPS) {
        double d = dist(pu, v) + dist(v, nu) - dist(pu, u) - dist(u, nu) +
                   dist(pv, u) + dist(u, nv) - dist(pv, v) - dist(v, nv);
        if (d < -EPS) {
          s->a[ru][iu] = v; s->a[rv][iv] = u;
          route_fix(s, ru); route_fix(s, rv);
          push(c, u); push(c, v); push(c, pu); push(c, nu); push(c, pv); push(c, nv);
          return 1;
        }
      }
      /* ---- (3) 2-opt* (tail exchange), four variants that create edge u-v ---- */
      int *b1 = c->buf, *b2 = c->buf2;
      int *A = s->a[ru], *B = s->a[rv];
      /* (A) A[..iu] + B[iv..]  |  B[..iv-1] + A[iu+1..] */
      if (preU + (LV - preVex) <= CAP + EPS && preVex + (LU - preU) <= CAP + EPS) {
        double d = dist(u, v) + dist(pv, nu) - dist(u, nu) - dist(pv, v);
        if (d < -EPS) {
          int n1 = 0, n2 = 0;
          for (int i = 0; i <= iu; i++) b1[n1++] = A[i];
          for (int i = iv; i < nV; i++) b1[n1++] = B[i];
          for (int i = 0; i < iv; i++) b2[n2++] = B[i];
          for (int i = iu + 1; i < nU; i++) b2[n2++] = A[i];
          set_two(s, ru, b1, n1, rv, b2, n2);
          push(c, u); push(c, v); push(c, pv); push(c, nu);
          return 1;
        }
      }
      /* (B) B[..iv] + A[iu..]  |  A[..iu-1] + B[iv+1..] */
      if (preV + (LU - preUex) <= CAP + EPS && preUex + (LV - preV) <= CAP + EPS) {
        double d = dist(v, u) + dist(pu, nv) - dist(pu, u) - dist(v, nv);
        if (d < -EPS) {
          int n1 = 0, n2 = 0;
          for (int i = 0; i <= iv; i++) b1[n1++] = B[i];
          for (int i = iu; i < nU; i++) b1[n1++] = A[i];
          for (int i = 0; i < iu; i++) b2[n2++] = A[i];
          for (int i = iv + 1; i < nV; i++) b2[n2++] = B[i];
          set_two(s, ru, b1, n1, rv, b2, n2);
          push(c, u); push(c, v); push(c, pu); push(c, nv);
          return 1;
        }
      }
      /* (C) A[..iu] + rev(B[..iv])  |  rev(A[iu+1..]) + B[iv+1..] */
      if (preU + preV <= CAP + EPS && (LU - preU) + (LV - preV) <= CAP + EPS) {
        double d = dist(u, v) + dist(nu, nv) - dist(u, nu) - dist(v, nv);
        if (d < -EPS) {
          int n1 = 0, n2 = 0;
          for (int i = 0; i <= iu; i++) b1[n1++] = A[i];
          for (int i = iv; i >= 0; i--) b1[n1++] = B[i];
          for (int i = nU - 1; i > iu; i--) b2[n2++] = A[i];
          for (int i = iv + 1; i < nV; i++) b2[n2++] = B[i];
          set_two(s, ru, b1, n1, rv, b2, n2);
          push(c, u); push(c, v); push(c, nu); push(c, nv);
          return 1;
        }
      }
      /* (D) rev(B[iv..]) + A[iu..]  |  A[..iu-1] + rev(B[..iv-1]) */
      if ((LV - preVex) + (LU - preUex) <= CAP + EPS && preUex + preVex <= CAP + EPS) {
        double d = dist(v, u) + dist(pu, pv) - dist(pu, u) - dist(pv, v);
        if (d < -EPS) {
          int n1 = 0, n2 = 0;
          for (int i = nV - 1; i >= iv; i--) b1[n1++] = B[i];
          for (int i = iu; i < nU; i++) b1[n1++] = A[i];
          for (int i = 0; i < iu; i++) b2[n2++] = A[i];
          for (int i = iv - 1; i >= 0; i--) b2[n2++] = B[i];
          set_two(s, ru, b1, n1, rv, b2, n2);
          push(c, u); push(c, v); push(c, pu); push(c, pv);
          return 1;
        }
      }
    } else {
      /* ---- (4) intra-route 2-opt creating edge u-v ---- */
      int i = iu < iv ? iu : iv, j = iu < iv ? iv : iu;
      if (j > i + 1) {
        int *A = s->a[ru];
        int x = A[i], y = A[j], x1 = A[i + 1], ny = AT(s, ru, j + 1), px = AT(s, ru, i - 1), py = A[j - 1];
        double d1 = dist(x, y) + dist(x1, ny) - dist(x, x1) - dist(y, ny); /* rev [i+1..j] */
        double d2 = dist(px, py) + dist(x, y) - dist(px, x) - dist(py, y); /* rev [i..j-1] */
        if (d1 < -EPS || d2 < -EPS) {
          if (d1 <= d2) rev(A, i + 1, j); else rev(A, i, j - 1);
          route_fix(s, ru);
          push(c, x); push(c, y); push(c, x1); push(c, ny); push(c, px); push(c, py);
          return 1;
        }
      }
    }
  }
  return 0;
}

static void local_search(sol_t *s, ls_t *c) {
  int u;
  while ((u = pop(c)) > 0)
    if (ls_try(s, c, u)) push(c, u);
}
static void push_all_random(ls_t *c) {
  int *perm = c->buf;
  for (int i = 1; i < N; i++) perm[i - 1] = i;
  for (int i = N - 2; i > 0; i--) { int j = rbound(&c->rng, i + 1); int t = perm[i]; perm[i] = perm[j]; perm[j] = t; }
  for (int i = 0; i < N - 1; i++) push(c, perm[i]);
}

/* ------------------------------------------------------------------------ */
/* Step 8: ruin (remove a cluster of nearby customers) and recreate          */
/* (cheapest insertion next to a nearby customer), then local search.       */
/* ------------------------------------------------------------------------ */
static void ruin_recreate(sol_t *s, ls_t *c) {
  int maxm = K < 30 ? K : 30;
  int m = 5 + (int)rbound(&c->rng, (uint32_t)(maxm - 4 > 1 ? maxm - 4 : 1));
  if (m > N - 1) m = N - 1;
  int seed = 1 + (int)rbound(&c->rng, N - 1);
  int *rem = xmalloc(sizeof(int) * (m + 1)), nrem = 0;
  rem[nrem++] = seed; c->mark[seed] = 1;
  const int *nb = KNN + (size_t)seed * K;
  for (int k = 0; k < K && nrem < m; k++)
    if (!c->mark[nb[k]]) { rem[nrem++] = nb[k]; c->mark[nb[k]] = 1; }
  /* remove from their routes */
  for (int i = 0; i < nrem; i++) {
    int r = s->rt[rem[i]];
    if (r < 0) continue;
    int n = 0, *A = s->a[r];
    for (int j = 0; j < s->len[r]; j++) if (!c->mark[A[j]]) A[n++] = A[j];
    s->len[r] = n; route_fix(s, r);
    for (int j = 0; j < nrem; j++) if (s->rt[rem[j]] == r) s->rt[rem[j]] = -1;
  }
  /* insertion order: random, or by decreasing demand, or by distance from depot */
  int mode = rbound(&c->rng, 3);
  for (int i = nrem - 1; i > 0; i--) { int j = rbound(&c->rng, i + 1); int t = rem[i]; rem[i] = rem[j]; rem[j] = t; }
  if (mode == 1) { /* demand desc (insertion sort, m is small) */
    for (int i = 1; i < nrem; i++) { int x = rem[i], j = i; while (j > 0 && DEM[rem[j-1]] < DEM[x]) { rem[j] = rem[j-1]; j--; } rem[j] = x; }
  } else if (mode == 2) { /* far from depot first */
    for (int i = 1; i < nrem; i++) { int x = rem[i], j = i; double dx = dist(0, x); while (j > 0 && dist(0, rem[j-1]) < dx) { rem[j] = rem[j-1]; j--; } rem[j] = x; }
  }
  for (int i = 0; i < nrem; i++) {
    int x = rem[i];
    c->mark[x] = 0;
    double bestc = 2 * dist(0, x); int br = -1, bj = -1;
    const int *nx = KNN + (size_t)x * K;
    for (int k = 0; k < K; k++) {
      int w = nx[k];
      if (c->mark[w] || s->rt[w] < 0) continue;  /* not routed yet */
      if (runif(&c->rng) < 0.01) continue;      /* blink */
      int r = s->rt[w];
      if (s->load[r] + DEM[x] > CAP + EPS) continue;
      int jw = s->ps[w];
      for (int t = 0; t < 2; t++) {
        int j = t ? jw - 1 : jw;                 /* insert after position j */
        int p = AT(s, r, j), q = AT(s, r, j + 1);
        double d = dist(p, x) + dist(x, q) - dist(p, q);
        if (d < bestc) { bestc = d; br = r; bj = j; }
      }
    }
    int *b = c->buf;
    if (br < 0) { br = sol_new_route(s); bj = -1; }
    int n = s->len[br];
    memcpy(b, s->a[br], sizeof(int) * (bj + 1));
    b[bj + 1] = x;
    memcpy(b + bj + 2, s->a[br] + bj + 1, sizeof(int) * (n - bj - 1));
    route_set(s, br, b, n + 1);
  }
  for (int i = 0; i < nrem; i++) {
    push(c, rem[i]);
    const int *nx = KNN + (size_t)rem[i] * K;
    for (int k = 0; k < 5 && k < K; k++) push(c, nx[k]);
  }
  free(rem);
  local_search(s, c);
}

/* parallel ruin-and-recreate with simulated-annealing acceptance.
 * Threads search independently and adopt the global best at sync points. */
static void ils(sol_t *best, double seconds, double *ils_cost) {
  double t0 = omp_get_wtime();
  double gbest = sol_cost(best);
  int nsync = seconds >= 4 ? 8 : 2;
  double avg_edge = gbest / (N - 1 + best->nr);
  long total_iters = 0;
#pragma omp parallel num_threads(P.nthreads) reduction(+ : total_iters)
  {
    int tid = omp_get_thread_num();
    sol_t cur, cand, mybest;
    sol_init(&cur); sol_init(&cand); sol_init(&mybest);
    ls_t c; ls_init(&c, P.seed * 7919 + 104729ULL * (tid + 1));
#pragma omp critical
    { sol_copy(&cur, best); }
    sol_copy(&mybest, &cur); sol_copy(&cand, &cur);
    double curc = sol_cost(&cur), myc = curc;
    for (int round = 0; round < nsync; round++) {
      double tend = t0 + seconds * (round + 1) / nsync;
      double now;
      while ((now = omp_get_wtime()) < tend) {
        /* temperature decreases linearly over the whole run */
        double frac = (now - t0) / seconds;
        double T = P.temp * avg_edge * (1.0 - frac) + 1e-6;
        ruin_recreate(&cand, &c);
        double cc = sol_cost(&cand);
        total_iters++;
        if (cc < curc - T * log(runif(&c.rng) + 1e-300)) {
          sol_copy(&cur, &cand); curc = cc;
          if (cc < myc - EPS) { sol_copy(&mybest, &cand); myc = cc; }
        } else sol_copy(&cand, &cur);
      }
      /* sync: publish my best, then everyone continues from the global best */
#pragma omp critical
      { if (myc < gbest - EPS) { gbest = myc; sol_copy(best, &mybest); } }
#pragma omp barrier
#pragma omp critical
      { if (gbest < myc - EPS) { sol_copy(&mybest, best); myc = gbest; } }
      sol_copy(&cur, &mybest); sol_copy(&cand, &mybest); curc = myc;
#pragma omp barrier
    }
    sol_free(&cur); sol_free(&cand); sol_free(&mybest); ls_free(&c);
  }
  sol_compact(best);
  *ils_cost = sol_cost(best);
  fprintf(stderr, "# ILS iterations: %ld in %.2fs\n", total_iters, omp_get_wtime() - t0);
}

/* ------------------------------------------------------------------------ */
/* Verification and output                                                  */
/* ------------------------------------------------------------------------ */
static int verify(const sol_t *s) {
  int *hist = xcalloc(N, sizeof(int)), ok = 1;
  for (int r = 0; r < s->nr; r++) {
    double l = 0;
    for (int i = 0; i < s->len[r]; i++) {
      int v = s->a[r][i];
      if (v <= 0 || v >= N) { ok = 0; continue; }
      hist[v]++; l += DEM[v];
    }
    if (l > CAP + EPS) ok = 0;
  }
  for (int v = 1; v < N; v++) if (hist[v] != 1) ok = 0;
  free(hist);
  return ok;
}

static void print_solution(FILE *f, const sol_t *s) {
  int k = 0;
  for (int r = 0; r < s->nr; r++) {
    if (!s->len[r]) continue;
    fprintf(f, "Route #%d:", ++k);
    for (int i = 0; i < s->len[r]; i++) fprintf(f, " %d", s->a[r][i]);
    fputc('\n', f);
  }
  double c = sol_cost(s);
  if (ROUND) fprintf(f, "Cost %.0f\n", c); else fprintf(f, "Cost %.2f\n", c);
}

/* ------------------------------------------------------------------------ */
static void usage(const char *prog) {
  fprintf(stderr,
    "cmds 1.0 - MDS heuristic for CVRP in C (OpenMP / pthreads)\n"
    "Usage: %s file.vrp [options]\n"
    "  -nthreads <n>     threads (default 2)\n"
    "  -round 0|1        round distances (default 1; use 0 for Golden/CMT)\n"
    "  -iters <rho>      randomized DFS iterations (default 100000)\n"
    "  -par omp|pthread  parallel runtime for the randomized phase (default omp)\n"
    "  -sched static|dynamic|guided|auto   loop schedule (default static)\n"
    "  -chunk <c>        chunk size for the schedule (default: runtime default)\n"
    "  -ls 0|1           0: original post-processing only, 1: + inter-route LS (default 1)\n"
    "  -starts <S>       improve the S best distinct MDS solutions (default 1)\n"
    "  -time <sec>       seconds of parallel ruin-and-recreate after LS (default 0)\n"
    "  -knn <k>          neighbourhood size for LS (default 40)\n"
    "  -matrix 0|1       precompute the distance matrix if <= 1 GiB (default 0)\n"
    "  -temp <f>         ILS start temperature, fraction of mean edge length (default 0.5)\n"
    "  -seed <s>         random seed (default 1)\n"
    "  -quiet            do not print the routes\n"
    "  -sol <file>       also write the solution to <file>\n", prog);
  exit(1);
}

int main(int argc, char **argv) {
  if (argc < 2) usage(argv[0]);
  for (int i = 2; i < argc; i++) {
    const char *a = argv[i];
    const char *v = i + 1 < argc ? argv[i + 1] : NULL;
#define NEEDV if (!v) usage(argv[0]); i++;
    if (!strcmp(a, "-nthreads")) { NEEDV P.nthreads = atoi(v); }
    else if (!strcmp(a, "-round")) { NEEDV ROUND = atoi(v); }
    else if (!strcmp(a, "-iters")) { NEEDV P.iters = atol(v); }
    else if (!strcmp(a, "-par")) { NEEDV P.par = !strcmp(v, "pthread") ? PAR_PTHREAD : PAR_OMP; }
    else if (!strcmp(a, "-sched")) {
      NEEDV
      if (!strcmp(v, "static")) P.sched = omp_sched_static;
      else if (!strcmp(v, "dynamic")) P.sched = omp_sched_dynamic;
      else if (!strcmp(v, "guided")) P.sched = omp_sched_guided;
      else if (!strcmp(v, "auto")) P.sched = omp_sched_auto;
      else usage(argv[0]);
    }
    else if (!strcmp(a, "-chunk")) { NEEDV P.chunk = atoi(v); }
    else if (!strcmp(a, "-ls")) { NEEDV P.ls = atoi(v); }
    else if (!strcmp(a, "-starts")) { NEEDV P.starts = atoi(v); }
    else if (!strcmp(a, "-time")) { NEEDV P.time = atof(v); }
    else if (!strcmp(a, "-knn")) { NEEDV P.knn = atoi(v); }
    else if (!strcmp(a, "-matrix")) { NEEDV P.matrix = atoi(v); }
    else if (!strcmp(a, "-temp")) { NEEDV P.temp = atof(v); }
    else if (!strcmp(a, "-seed")) { NEEDV P.seed = strtoull(v, NULL, 10); }
    else if (!strcmp(a, "-quiet")) { P.quiet = 1; }
    else if (!strcmp(a, "-sol")) { NEEDV P.solfile = v; }
    else usage(argv[0]);
  }
  if (P.nthreads < 1) P.nthreads = 1;
  if (P.iters < 1) P.iters = 1;
  if (P.starts < 1) P.starts = 1;
  if (P.ls == 0) { P.starts = 1; P.time = 0; }

  read_vrp(argv[1]);
  ND = xmalloc(sizeof(node_t) * N);
  for (int i = 0; i < N; i++) ND[i] = (node_t){X[i], Y[i], DEM[i], 0};
  double t_start = omp_get_wtime();
  if (P.matrix) build_matrix();

  /* Step 1: MST */
  int *par = prim_mst();
  int *off, *adj;
  mst_csr(par, &off, &adj);
  free(par);
  G_OFF = off; G_ADJ = adj;

  /* iteration 0: plain DFS of the MST  (reported as "Step 1" like the original) */
  pool_t pool; pool_init(&pool, P.starts);
  {
    work_t w; work_alloc(&w);
    double c0 = mds_iter(0, &w, DBL_MAX);
    pool_add(&pool, c0, 0, w.order);
    work_free(&w);
  }
  double cost1 = pool.cost[0], t1 = omp_get_wtime() - t_start;

  /* Steps 2-4: randomized DFS iterations */
  if (P.iters > 1) {
    if (P.par == PAR_PTHREAD) random_phase_pthread(&pool);
    else random_phase_omp(&pool);
  }
  double cost2 = pool.cost[0], t2 = omp_get_wtime() - t_start;

  /* Step 5 (+6, 7): post-process (and local search) every start, keep the best */
  int S = pool.cnt;
  sol_t *sols = xmalloc(sizeof(sol_t) * S);
  double *pc = xmalloc(sizeof(double) * S), *lc = xmalloc(sizeof(double) * S);
  int outer = S > 1 ? (S < P.nthreads ? S : P.nthreads) : 1;
  int inner = S > 1 ? 1 : P.nthreads;
#pragma omp parallel for schedule(dynamic, 1) num_threads(outer)
  for (int k = 0; k < S; k++) {
    sol_init(&sols[k]);
    sol_from_tour(&sols[k], pool.tour[k]);
    postprocess(&sols[k], inner);
    pc[k] = sol_cost(&sols[k]);
    lc[k] = pc[k];
  }
  double t3 = omp_get_wtime() - t_start;
  if (P.ls) {
    build_knn();
#pragma omp parallel for schedule(dynamic, 1) num_threads(outer)
    for (int k = 0; k < S; k++) {
      ls_t c; ls_init(&c, P.seed + 31 * k);
      push_all_random(&c);
      local_search(&sols[k], &c);
      /* finish with a full 2-opt pass on each route */
      for (int r = 0; r < sols[k].nr; r++) { two_opt(sols[k].a[r], sols[k].len[r]); route_fix(&sols[k], r); }
      sol_compact(&sols[k]);
      lc[k] = sol_cost(&sols[k]);
      ls_free(&c);
    }
  }
  double t4 = omp_get_wtime() - t_start;
  int bk = 0;
  for (int k = 1; k < S; k++) if (lc[k] < lc[bk] - EPS) bk = k;
  /* cost3: original pipeline's result (post-processing of the best MDS tour) */
  double cost3 = pc[0];
  sol_t *best = &sols[bk];
  double cost4 = lc[bk];

  /* Step 8: time-bounded parallel ruin-and-recreate */
  double cost5 = cost4;
  if (P.ls && P.time > 0) ils(best, P.time, &cost5);
  double t5 = omp_get_wtime() - t_start;

  int ok = verify(best);
  double final = sol_cost(best);
  if (fabs(final - (P.time > 0 ? cost5 : (P.ls ? cost4 : cost3))) > 1e-6) ok = 0;

  fprintf(stderr, "%s Cost %.2f %.2f %.2f", argv[1], cost1, cost2, cost3);
  if (P.ls) fprintf(stderr, " LS %.2f", cost4);
  if (P.time > 0) fprintf(stderr, " ILS %.2f", cost5);
  fprintf(stderr, " Time(seconds) %.6f %.6f %.6f", t1, t2, t3);
  if (P.ls) fprintf(stderr, " %.6f", t4);
  if (P.time > 0) fprintf(stderr, " %.6f", t5);
  fprintf(stderr, " threads %d routes %d %s\n", P.nthreads, best->nr, ok ? "VALID" : "INVALID");

  if (!P.quiet) print_solution(stdout, best);
  if (P.solfile) {
    FILE *f = fopen(P.solfile, "w");
    if (f) { print_solution(f, best); fclose(f); }
  }
  for (int k = 0; k < S; k++) sol_free(&sols[k]);
  free(sols); free(pc); free(lc);
  pool_free(&pool);
  free(off); free(adj);
  return ok ? 0 : 2;
}
