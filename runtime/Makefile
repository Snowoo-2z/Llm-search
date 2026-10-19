CC ?= cc
CFLAGS ?= -O3 -march=native -funroll-loops -ffast-math -fopenmp -std=c99 -Wall -Wextra
LDFLAGS ?= -fopenmp
bcrun: bcrun.c
	$(CC) $(CFLAGS) -o bcrun bcrun.c -lm
clean:
	rm -f bcrun
.PHONY: clean
