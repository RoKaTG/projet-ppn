#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matrix.h"


Matrix* create_matrix(int rows, int columns) {
    Matrix* matrix = (Matrix*)malloc(sizeof(Matrix));
    if (matrix != NULL) {
        matrix->values = (double**)malloc(sizeof(double*));
        matrix->rows = rows;
        matrix -> columns = columns;
        if (matrix->values != NULL) {
            for(int i = 0; i < row; i++) {
                matrix->values[i] = (double*)malloc(col * sizeof(double));
            }
        }
    }
    return matrix;
}

Matrix* copy_matrix(Matrix *original) {
    if (original == NULL) {
        return NULL;  // Retourne NULL si la matrice originale est NULL
    }

    // Créer une nouvelle matrice avec les mêmes dimensions que l'originale
    Matrix* new_matrix = create_matrix(original->row, original->column);
    if (new_matrix == NULL) {
        return NULL;  // Retourne NULL si l'allocation de mémoire échoue
    }

    // Copier les valeurs de l'originale dans la nouvelle matrice
    for (int i = 0; i < original->row; i++) {
        for (int j = 0; j < original->column; j++) {
            new_matrix->value[i][j] = original->value[i][j];
        }
    }

    return new_matrix;
}
