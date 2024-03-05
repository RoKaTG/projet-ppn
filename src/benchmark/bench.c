#include <cblas.h>
#include <clapack.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <stdint.h>
#include <omp.h>

#include "../../include/benchmark/bench.h"

void printBenchmarkResult(const Benchmark *result) {
    printf("%-10s %-20s %-10s %-20d %-10d %-15d %-15.2f %-20.2f %-20.2f %-15.2f\n", 
        result->routine, result->topology, result->actFunction, result->trainingImages, 
        result->epochs, result->batchSize, result->precision, result->totalTime, 
        result->avgEpochTime, result->errorRate);
}
