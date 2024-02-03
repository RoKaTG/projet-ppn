#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "matrix.h"
/**************************************/
/*                                    */
/*         Matrix's functions         */
/*                                    */
/**************************************/

Matrix* create_matrix(int r, int c) {
    Matrix* mat = (Matrix*)malloc(sizeof(Matrix));
    
    if (mat == NULL) {
        fprintf(stderr, "Memory allocation error for the matrix.\n");
        exit(1);
    }
    
    mat->row = r;
    mat->col = c;
    
    mat->val = (double**)malloc(r * sizeof(double*));
    
    if (mat->val == NULL) {
        fprintf(stderr, "Memory allocation error for matrix rows.\n");
        free(mat);
        exit(1);
    }
    
    for (int i = 0; i < r; i++) {
        mat->val[i] = (double*)malloc(c * sizeof(double));
        
        if (mat->val[i] == NULL) {
            fprintf(stderr, "Memory allocation error for matrix columns.\n");
            for (int j = 0; j < i; j++) {
                free(mat->val[j]);
            }
            free(mat->val);
            free(mat);
            exit(1);
        }
        
        // Initialize matrix values to 0.0
        for (int j = 0; j < c; j++) {
            mat->val[i][j] = 0.0;
        }
    }
    
    return mat;
}

void fill_matrix(Matrix* mat, double n) {
    for (int i = 0; i < mat->row; i++) {
        for (int j = 0; j < mat->col; j++) {
            mat->val[i][j] = n;
        }
    }
}

void matrix_randomize(Matrix* mat) {
    srand(time(NULL)); // Initialisation de la graine pour rand()
    
    for (int i = 0; i < mat->row; i++) {
        for (int j = 0; j < mat->col; j++) {
            mat->val[i][j] = (double)rand() / RAND_MAX; // Valeurs aléatoires entre 0 et 1
        }
    }
}

Matrix* matrix_copy(Matrix* mat) {
    Matrix* copy = create_matrix(mat->row, mat->col);
    
    for (int i = 0; i < mat->row; i++) {
        for (int j = 0; j < mat->col; j++) {
            copy->val[i][j] = mat->val[i][j];
        }
    }
    
    return copy;
}

void free_matrix(Matrix* mat) {
    for (int i = 0; i < mat->row; i++) {
        free(mat->val[i]);
    }
    free(mat->val);
    free(mat);
}

/**************************************/
/*                                    */
/*      Operation's functions         */
/*                                    */
/**************************************/

void scale_matrix(Matrix* mat, double scalar) {
    for (int i = 0; i < mat->row; i++) {
        for (int j = 0; j < mat->col; j++) {
            mat->val[i][j] *= scalar;
        }
    }
}

Matrix* matrix_dgemm(Matrix* mat1, Matrix* mat2) {
    if (mat1->col != mat2->row) {
        fprintf(stderr, "Error: Incompatible matrix dimensions for multiplication.\n");
        exit(1);
    }

    int resultRows = mat1->row;
    int resultCols = mat2->col;
    Matrix* result = create_matrix(resultRows, resultCols);

    for (int i = 0; i < resultRows; i++) {
        for (int j = 0; j < resultCols; j++) {
            for (int k = 0; k < mat1->col; k++) {
                result->val[i][j] += mat1->val[i][k] * mat2->val[k][j];
            }
        }
    }

    return result;
}

Matrix* matrix_add_vector(Matrix* mat, Matrix* vector) {
    if (vector->row != mat->row || vector->col != 1) {
        fprintf(stderr, "Error: Incompatible dimensions for matrix and vector addition.\n");
        exit(1);
    }

    Matrix* result = create_matrix(mat->row, mat->col);

    for (int i = 0; i < mat->row; i++) {
        for (int j = 0; j < mat->col; j++) {
            result->val[i][j] = mat->val[i][j] + vector->val[i][0];
        }
    }

    return result;
}

Matrix* matrix_subtract(Matrix* mat1, Matrix* mat2) {
    if (mat1->row != mat2->row || mat1->col != mat2->col) {
        fprintf(stderr, "Error: Incompatible matrix dimensions for subtraction.\n");
        exit(1);
    }

    Matrix* result = create_matrix(mat1->row, mat1->col);

    for (int i = 0; i < mat1->row; i++) {
        for (int j = 0; j < mat1->col; j++) {
            result->val[i][j] = mat1->val[i][j] - mat2->val[i][j];
        }
    }

    return result;
}

Matrix* matrix_transpose(Matrix* mat) {
    Matrix* result = create_matrix(mat->col, mat->row);

    for (int i = 0; i < mat->row; i++) {
        for (int j = 0; j < mat->col; j++) {
            result->val[j][i] = mat->val[i][j];
        }
    }

    return result;
}

