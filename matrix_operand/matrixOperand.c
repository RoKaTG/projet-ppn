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
