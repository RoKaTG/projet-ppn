#ifndef MATRIX_H
#define MATRIX_H

typedef struct {
    int row;
    int column;
    double** value;
} Matrix;

void print_matrix(Matrix *matrix);

#endif /* MATRIX_H */
