#pragma once

typedef struct {
    char* routine; // "batch" ou "default"
    char* topology; // Par exemple, "784,128,10"
    char* actFunction; // "relu", "sigmoid", ou "tanh"
    int trainingImages;
    int epochs;
    int batchSize; // -1 si non applicable
    float precision;
    double totalTime;
    double avgEpochTime;
    float errorRate;
} Benchmark;

void printBenchmarkResult(const Benchmark *result);