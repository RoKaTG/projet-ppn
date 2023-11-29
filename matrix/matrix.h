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
Matrix* create_matrix(int r, int c);
void free_matrix(Matrix** matrix);
double gaussian_random(double mean, double std_dev);
void randomize_matrix(Matrix* m, double mean, double std_dev);

#endif /* MATRIX_H */
