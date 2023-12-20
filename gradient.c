#include <stdio.h>


double fonction(double x) {
    return x * x;
}

double calculer_derivee_numerique(double (*fonction)(double), double x, double h) {
    return (fonction(x + h) - fonction(x)) / h;
}

void descente_gradient(double taux_apprentissage, int iterations, double h) {
    double x = 0.0;
    for (int i = 0; i < iterations; ++i) {
        double gradient = calculer_derivee_numerique(fonction, x, h); 
        x = x - taux_apprentissage * gradient;
        printf("ItÃ©ration %d : x = %lf, f(x) = %lf\n", i + 1, x, fonction(x));
    }
}

int main() {
    double taux_apprentissage = 0.1; 
    int iterations = 100; 
    double h = 0.0001; 

    descente_gradient(taux_apprentissage, iterations, h);

    return 0;
}

