CC      ?= gcc
CFLAGS  ?= -O3 -march=native -fopenmp -Wall -Wextra
LDLIBS  = -lm -lpthread
INPUTS  ?= ../parMDS/inputs

all: cmds

cmds: cmds.c
	$(CC) $(CFLAGS) cmds.c -o cmds $(LDLIBS)

# same code without SIMD (for the SIMD experiment)
cmds_nosimd: cmds.c
	$(CC) -O3 -march=native -fopenmp -fno-openmp-simd -fno-tree-vectorize cmds.c -o cmds_nosimd $(LDLIBS)

test: cmds
	./cmds toy.vrp -ls 0
	./cmds $(INPUTS)/X-n101-k25.vrp -quiet -time 3

clean:
	rm -f cmds cmds_nosimd

.PHONY: all test clean
