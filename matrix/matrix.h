#ifndef MATRIX_H
#define MATRIX_H

typedef struct {
    int row;
    int column;
    double** value;
} Matrix;

Matrix* create_matrix(int r, int c);

#endif /* MATRIX_H */
