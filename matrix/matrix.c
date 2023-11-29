#include <stdio.h>
#include <stdlib.h>

#include "matrix.h"


Matrix* create_matrix(int r, int c) {
    if (r <= 0 || c <= 0) {
        return NULL;
    }
    Matrix* matrix = (Matrix*)malloc(sizeof(Matrix));
    if (matrix == NULL) {
        return NULL;
    }

    matrix->row = r;
    matrix->column = c;
    matrix->value = (double**)malloc(r * sizeof(double*));

    if (matrix->value == NULL) {
        free(matrix);
        return NULL;
    }

    for (int i = 0; i < r; i++) {
        matrix->value[i] = (double*)malloc(c * sizeof(double));
        if (matrix->value[i] == NULL) {
            for (int j = 0; j < i; j++) {
                free(matrix->value[j]);
            }
            free(matrix->value);
            free(matrix);
            return NULL;
        }
    }

    return matrix;
}

int fill_matrix(Matrix *matrix, double values[], int values_size) {
    if (matrix == NULL || values == NULL) {
        return -1;
    }

    int needed_size = matrix->row * matrix->column;
    if (values_size < needed_size) {
        return -1;
    }

    for (int i = 0; i < matrix->row; i++) {
        for (int j = 0; j < matrix->column; j++) {
            matrix->value[i][j] = values[i * matrix->column + j];
        }
    }
    return 0;
}

void print_matrix(Matrix *matrix) {
    if (matrix == NULL) {
        printf("Matrix is NULL\n");
        return;
    }

    printf("This is the content of this %d x %d matrix:\n", matrix->row, matrix->column);
    for (int i = 0; i < matrix->row; i++) {
        for (int j = 0; j < matrix->column; j++) {
            printf("%lf \t", matrix->value[i][j]);
        }
        printf("\n");
    }
}

void free_matrix(Matrix** matrix) {
    if (*matrix == NULL) {
        return;
    }

    for (int i = 0; i < (*matrix)->row; i++) {
        free((*matrix)->value[i]);
    }
    free((*matrix)->value);
    free(*matrix);

    *matrix = NULL;
}

Size check_dimensions(Matrix *matrix) {
    Size size;

    if (matrix == NULL) {
        size.rows = 0;
        size.columns = 0;
        size.type = MATRIX_TYPE;
        return size;
    }

    size.rows = matrix->row;
    size.columns = matrix->column;

    if (matrix->column == 1) {
        size.type = VECTOR_TYPE;
    } else {
        size.type = MATRIX_TYPE;
    }

    return size;
}

Matrix* copy_matrix(Matrix *original) {
    if (original == NULL) {
        return NULL;  
    }

    Matrix* new_matrix = create_matrix(original->row, original->column);
    if (new_matrix == NULL) {
        return NULL; 
    }

    for (int i = 0; i < original->row; i++) {
        for (int j = 0; j < original->column; j++) {
            new_matrix->value[i][j] = original->value[i][j];
        }
    }

    return new_matrix;
}

int save_matrix(Matrix *matrix, const char *filename) {
    if (matrix == NULL || filename == NULL) {
        return -1;  
    }

    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        return -1;  
    }

    fprintf(file, "%d %d\n", matrix->row, matrix->column);

    for (int i = 0; i < matrix->row; i++) {
        for (int j = 0; j < matrix->column; j++) {
            fprintf(file, "%lf ", matrix->value[i][j]);
        }
        fprintf(file, "\n");
    }

    fclose(file);
    return 0; 
}

Matrix* load_matrix(const char *filename) {
    if (filename == NULL) {
        return NULL;
    }

    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        return NULL;
    }

    int rows, columns;
    if (fscanf(file, "%d %d\n", &rows, &columns) != 2) {
        fclose(file);
        return NULL;  
    }

    Matrix *matrix = create_matrix(rows, columns);
    if (matrix == NULL) {
        fclose(file);
        return NULL; 
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            if (fscanf(file, "%lf", &matrix->value[i][j]) != 1) {
                fclose(file);
                free_matrix(&matrix);
                return NULL; 
            }
        }
    }

    fclose(file);
    return matrix;
}

double gaussian_random(double mean, double std_dev) {
    double u = ((double) rand() / (RAND_MAX)) * 2 - 1;
    double v = ((double) rand() / (RAND_MAX)) * 2 - 1;
    double r = u * u + v * v;

    if (r == 0 || r > 1) {
        return gaussian_random(mean, std_dev);
    }

    double c = sqrt(-2 * log(r) / r);
    return mean + u * c * std_dev;
}

void matrix_randomize(Matrix* m, double mean, double std_dev) {
    if (m == NULL) {
        return;
    }

    for (int i = 0; i < m->row; i++) {
        for (int j = 0; j < m->column; j++) {
            m->value[i][j] = gaussian_random(mean, std_dev);
        }
    }
}

