#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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
