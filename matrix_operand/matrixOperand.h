#ifndef MATRIXOPERAND_H
#define MATRIXOPERAND_H

#include "../matrix/matrix.h"

void scale_matrix(Matrix* matrix, double scalar);
Matrix* inverse_matrix(Matrix* matrix);
Matrix* add_matrix(Matrix* matrix1, Matrix* matrix2);
Matrix* sub_matrix(Matrix* matrix1, Matrix* matrix2);

#endif //MATRIXOPERAND_H
