#pragma once

typedef struct {
    char* routine; 
    char* topology; 
    char* actFunction; 
    int trainingImages;
    int epochs;
    int batchSize;
    float precision;
    double totalTime;
    double avgEpochTime;
    float errorRate;
} Benchmark;

void printBenchmarkResult(const Benchmark *result);