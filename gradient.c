#include <stdio.h>


double fonction(double x) {
    return x * x;
}

double calculer_derivee_numerique(double (*fonction)(double), double x, double h) {
    return (fonction(x + h) - fonction(x)) / h;
}
