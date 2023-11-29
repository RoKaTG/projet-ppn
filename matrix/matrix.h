#ifndef MATRIX_H
#define MATRIX_H

typedef struct {
    int row;
    int column;
    double** value;
} Matrix;

Matrix* copy_matrix(Matrix *original);
int save_matrix(Matrix *matrix, const char *filename);
Matrix* load_matrix(const char *filename);

#endif /* MATRIX_H */
