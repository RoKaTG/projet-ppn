#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>

#include "matrixOperand.h"

void scale_matrix(Matrix* matrix, double scalar) {
    if (matrix == NULL || matrix->value == NULL) return;

    for (int i = 0; i < matrix->row; i++) {
        for (int j = 0; j < matrix->column; j++) {
            matrix->value[i][j] *= scalar;
        }
    }
}

Matrix* inverse_matrix(Matrix* matrix) {
    return NULL;
}


Matrix* add_matrix(Matrix* matrix1, Matrix* matrix2) {
    if (matrix1 == NULL || matrix2 == NULL || matrix1->row != matrix2->row || matrix1->column != matrix2->column) {
        return NULL;
    }

    Matrix* result = create_matrix(matrix1->row, matrix1->column);
    if (result == NULL) {
        return NULL;
    }

    for (int i = 0; i < matrix1->row; i++) {
        for (int j = 0; j < matrix1->column; j++) {
            result->value[i][j] = matrix1->value[i][j] + matrix2->value[i][j];
        }
    }

    return result;
}

Matrix* sub_matrix(Matrix* matrix1, Matrix* matrix2) {
    if (matrix1 == NULL || matrix2 == NULL || matrix1->row != matrix2->row || matrix1->column != matrix2->column) {
        return NULL;
    }

    Matrix* result = create_matrix(matrix1->row, matrix1->column);
    if (result == NULL) {
        return NULL;
    }

    for (int i = 0; i < matrix1->row; i++) {
        for (int j = 0; j < matrix1->column; j++) {
            result->value[i][j] = matrix1->value[i][j] - matrix2->value[i][j];
        }
    }

    return result;
}

Matrix* dotprod(Matrix *matrix, Matrix *vector) {
    if (matrix == NULL || vector == NULL || vector->column != 1 || matrix->column != vector->row) {
        return NULL; 
    }

    int result_matrix_rows = matrix->row;

    Matrix *result_vector = create_matrix(result_matrix_rows, 1);
    if (result_vector == NULL) {
        return NULL; 
    }

    for (int i = 0; i < result_matrix_rows; i++) {
        result_vector->value[i][0] = 0;
        for (int k = 0; k < matrix->column; k++) {
            result_vector->value[i][0] += matrix->value[i][k] * vector->value[k][0];
        }
    }

    return result_vector;
}

bool compare_matrix(Matrix* matrix1, Matrix* matrix2) {
    if (matrix1 == NULL || matrix2 == NULL) {
        printf("L'une des matrices est NULL.\n");
        return false;
    }

    if (matrix1->row == matrix2->row && matrix1->column == matrix2->column) {
        printf("Les matrices sont de mêmes dimensions.\n");
        return true;
    } else {
        printf("Les matrices ne sont pas de mêmes dimensions.\n");
        return false;
    }
}
