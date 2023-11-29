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
