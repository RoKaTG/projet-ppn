#ifndef MATRIX_H
#define MATRIX_H

typedef struct {
    int row;
    int column;
    double** value;
} Matrix;

Matrix* create_matrix(int r, int c);
int fill_matrix(Matrix *matrix, double values[], int values_size);
void free_matrix(Matrix** matrix);
Matrix* copy_matrix(Matrix *original);
int save_matrix(Matrix *matrix, const char *filename);
Matrix* load_matrix(const char *filename);
double gaussian_random(double mean, double std_dev);
void matrix_randomize(Matrix* m, double mean, double std_dev);

#endif /* MATRIX_H */
