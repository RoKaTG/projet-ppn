#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "matrixOperand.h"

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

