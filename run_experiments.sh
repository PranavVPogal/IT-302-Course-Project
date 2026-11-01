#!/bin/bash
# Runs all experiments serially (so timings don't interfere). Results go to results/*.csv
cd "$(dirname "$0")"
R=results; mkdir -p $R
# wait for the original-code baseline to finish
while pgrep -f "bench.sh orig_seq" > /dev/null; do sleep 20; done

# E1: C port, original pipeline only (-ls 0), 1 and 2 threads, all instances
./bench.sh c_seq  $R/c_seq.csv  all.txt ./cmds -quiet -ls 0 -nthreads 1
./bench.sh c_par2 $R/c_par2.csv all.txt ./cmds -quiet -ls 0 -nthreads 2

# E2: scheduling / runtime experiments on a size-spread subset (-ls 0, 2 threads)
for cfg in "omp static 0" "omp static 1" "omp static 64" "omp dynamic 1" "omp dynamic 64" "omp dynamic 1024" \
           "omp guided 0" "omp guided 64" "omp auto 0" "pthread static 0" "pthread dynamic 64"; do
  set -- $cfg
  ./bench.sh "$1-$2-$3" $R/sched.csv subset.txt ./cmds -quiet -ls 0 -nthreads 2 -par $1 -sched $2 -chunk $3
done
# SIMD on/off (Prim + matrix loops), and the precomputed distance matrix
./bench.sh "nosimd" $R/simd.csv subset.txt ./cmds_nosimd -quiet -ls 0 -nthreads 2 -iters 1
./bench.sh "simd"   $R/simd.csv subset.txt ./cmds        -quiet -ls 0 -nthreads 2 -iters 1
./bench.sh "matrix" $R/sched.csv subset.txt ./cmds -quiet -ls 0 -nthreads 2 -matrix 1
# original parMDS built with g++ -fopenmp (2 threads), for reference
./bench.sh "orig_par2" $R/orig_par2.csv subset.txt ./orig_par -nthreads 2

# E3: + inter-route local search, all instances
./bench.sh c_ls $R/c_ls.csv all.txt ./cmds -quiet -ls 1 -nthreads 2
# E4: + 10 s parallel ruin-and-recreate, all instances
./bench.sh c_ils10 $R/c_ils10.csv all.txt ./cmds -quiet -ls 1 -nthreads 2 -time 10
# E5: how many MDS iterations are worth it once LS+ILS are on? (subset)
for it in 1000 10000 100000; do
  ./bench.sh "iters$it" $R/iters.csv subset.txt ./cmds -quiet -ls 1 -nthreads 2 -time 10 -iters $it
done
echo ALLDONE > $R/DONE
