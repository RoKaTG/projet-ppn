#pragma once

/**************************************/
/*                                    */
/*          Matrix stuctures          */
/*                                    */
/**************************************/

typedef struct {
	int row;
	int col;
	double** val;
} Matrix;

/**************************************/
/*                                    */
/*         Matrix's headers           */
/*                                    */
/**************************************/

Matrix* create_matrix(int r, int c);
void fill_matrix(Matrix* mat, double n);
void matrix_randomize(Matrix* mat);
Matrix* matrix_copy(Matrix* mat);
void free_matrix(Matrix* mat);

/**************************************/
/*                                    */
/*        Operation's headers         */
/*                                    */
/**************************************/

void scale_matrix(Matrix* mat, double scalar);
Matrix* matrix_dgemm(Matrix* mat1, Matrix* mat2);
Matrix* matrix_add_vector(Matrix* mat, Matrix* vector);
Matrix* matrix_subtract(Matrix* mat1, Matrix* mat2);
Matrix* matrix_transpose(Matrix* mat);