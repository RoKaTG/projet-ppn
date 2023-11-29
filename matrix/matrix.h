#ifndef MATRIX_H
#define MATRIX_H

typedef struct {
    int row;
    int column;
    double** value;
} Matrix;

Matrix* copy_matrix(Matrix *original);

#endif /* MATRIX_H */
