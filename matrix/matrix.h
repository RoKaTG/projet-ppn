#ifndef MATRIX_H
#define MATRIX_H

typedef struct {
    int row;
    int column;
    double** value;
} Matrix;

int fill_matrix(Matrix *matrix, double values[], int values_size);

#endif /* MATRIX_H */
