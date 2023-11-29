#ifndef MATRIX_H
#define MATRIX_H

typedef struct {
    int row;
    int column;
    double** value;
} Matrix;

Matrix* create_matrix(int r, int c);
void free_matrix(Matrix** matrix);
double gaussian_random(double mean, double std_dev);
void randomize_matrix(Matrix* m, double mean, double std_dev);

#endif /* MATRIX_H */
