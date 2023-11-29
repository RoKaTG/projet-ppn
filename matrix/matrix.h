#ifndef MATRIX_H
#define MATRIX_H

typedef struct {
    int row;
    int column;
    double** value;
} Matrix;

typedef struct {
    int rows;
    int columns;
    MatrixType type;
} Size;

typedef enum {
    MATRIX_TYPE,
    VECTOR_TYPE
} MatrixType;

void print_matrix(Matrix *matrix);
Size check_dimensions(Matrix *matrix);

#endif /* MATRIX_H */
