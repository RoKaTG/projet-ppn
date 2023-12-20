#ifndef GRADIENT_H
#define GRADIENT_H

double fonction(double x);
double calculer_derivee_numerique(double (*fonction)(double), double x, double h);
void descente_gradient(double taux_apprentissage, int iterations, double h);

#endif /* GRADIENT_H */
