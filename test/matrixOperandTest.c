#include <stdint.h>
#include <setjmp.h>
#include <stdarg.h>
#include <stddef.h>
#include <cmocka.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#include "../matrix_operand/matrixOperand.h"

/******************************************/    

static void test_add_matrix(void **state) {
    Matrix *m1 = create_matrix(2, 2);
    Matrix *m2 = create_matrix(2, 2);
    double vals1[] = {1.0, 2.0, 3.0, 4.0};
    double vals2[] = {4.0, 3.0, 2.0, 1.0};
    fill_matrix(m1, vals1, 4);
    fill_matrix(m2, vals2, 4);

    Matrix *result = add_matrix(m1, m2);
    assert_non_null(result);

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            assert_float_equal(result->value[i][j], vals1[i * 2 + j] + vals2[i * 2 + j], 0.0001);
        }
    }

    Matrix *m3 = create_matrix(3, 2);
    Matrix *result_diff_size = add_matrix(m1, m3);
    assert_null(result_diff_size);

    Matrix *result_null = add_matrix(m1, NULL);
    assert_null(result_null);

    free_matrix(&m1);
    free_matrix(&m2);
    free_matrix(&m3);
    free_matrix(&result);
}

/******************************************/    

static void test_sub_matrix(void **state) {
    Matrix *m1 = create_matrix(2, 2);
    Matrix *m2 = create_matrix(2, 2);
    double vals1[] = {5.0, 7.0, 9.0, 11.0};
    double vals2[] = {1.0, 2.0, 3.0, 4.0};
    fill_matrix(m1, vals1, 4);
    fill_matrix(m2, vals2, 4);

    Matrix *result = sub_matrix(m1, m2);
    assert_non_null(result);

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            assert_float_equal(result->value[i][j], vals1[i * 2 + j] - vals2[i * 2 + j], 0.0001);
        }
    }

    Matrix *m3 = create_matrix(3, 2);
    Matrix *result_diff_size = sub_matrix(m1, m3);
    assert_null(result_diff_size);

    Matrix *result_null = sub_matrix(m1, NULL);
    assert_null(result_null);

    free_matrix(&m1);
    free_matrix(&m2);
    free_matrix(&m3);
    free_matrix(&result);
}

/******************************************/    

static void test_scale_matrix(void **state) {
    Matrix *m = create_matrix(2, 2);
    assert_non_null(m); 
    double vals[] = {1.0, 2.0, 3.0, 4.0};
    int fill_status = fill_matrix(m, vals, 4);
    assert_int_equal(fill_status, 0); 

    scale_matrix(m, 2.0);
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            assert_float_equal(m->value[i][j], vals[i * 2 + j] * 2.0, 0.0001);
        }
    }

    scale_matrix(m, 0.0);
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            assert_float_equal(m->value[i][j], 0.0, 0.0001);
        }
    }

    fill_matrix(m, vals, 4);

    scale_matrix(m, -1.0);
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            assert_float_equal(m->value[i][j], -vals[i * 2 + j], 0.0001);
        }
    }

    Matrix *empty = create_matrix(0, 0);
    assert_null(empty); 
    scale_matrix(empty, 2.0);
    
    scale_matrix(NULL, 2.0); 

    free_matrix(&m);
    free_matrix(&empty);
}

/******************************************/    

static void test_dotprod(void **state) {
    Matrix *m = create_matrix(3, 3);
    double vals_m[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    fill_matrix(m, vals_m, 9);

    Matrix *v = create_matrix(3, 1);
    double vals_v[] = {1, 2, 3};
    fill_matrix(v, vals_v, 3);

    Matrix *result = dotprod(m, v);
    assert_non_null(result);
    assert_int_equal(result->row, 3);
    assert_int_equal(result->column, 1);
    assert_float_equal(result->value[0][0], 14, 0.0001);
    assert_float_equal(result->value[1][0], 32, 0.0001);
    assert_float_equal(result->value[2][0], 50, 0.0001);
    free_matrix(&result);

    Matrix *m_incompatible = create_matrix(2, 3);
    Matrix *v_incompatible = create_matrix(4, 1);
    result = dotprod(m_incompatible, v_incompatible);
    assert_null(result);
    free_matrix(&m_incompatible);
    free_matrix(&v_incompatible);

    Matrix *v_empty = create_matrix(0, 0);
    result = dotprod(m, v_empty);
    assert_null(result);
    free_matrix(&v_empty);

    result = dotprod(m, NULL);
    assert_null(result);

    Matrix *v_1xN = create_matrix(1, 3);
    result = dotprod(m, v_1xN);
    assert_null(result);
    free_matrix(&v_1xN);

    free_matrix(&m);
    free_matrix(&v);
}

/******************************************/

static void test_compare_matrix(void **state) {
    Matrix *m1 = create_matrix(3, 3);
    Matrix *m2 = create_matrix(3, 3);
    Matrix *m3 = create_matrix(2, 3);
    Matrix *m4 = NULL;

    bool same = compare_matrix(m1, m2);
    assert_true(same);

    same = compare_matrix(m1, m3);
    assert_false(same);

    same = compare_matrix(m1, m4);
    assert_false(same);

    same = compare_matrix(m4, m4);
    assert_false(same);

    free_matrix(&m1);
    free_matrix(&m2);
    free_matrix(&m3);
}

/******************************************/

static void test_dgemm(void **state) {
    Matrix *m1 = create_matrix(2, 3);
    Matrix *m2 = create_matrix(3, 2);
    double vals_m1[] = {1, 2, 3, 4, 5, 6};
    double vals_m2[] = {7, 8, 9, 10, 11, 12};
    fill_matrix(m1, vals_m1, 6);
    fill_matrix(m2, vals_m2, 6);

    Matrix *result = dgemm(m1, m2);
    assert_non_null(result);
    assert_int_equal(result->row, 2);
    assert_int_equal(result->column, 2);

    assert_float_equal(result->value[0][0], 58, 0.0001);
    assert_float_equal(result->value[0][1], 64, 0.0001);
    assert_float_equal(result->value[1][0], 139, 0.0001);
    assert_float_equal(result->value[1][1], 154, 0.0001);
    
    Matrix *matrix1 = create_matrix(3, 3);
    Matrix *vector = create_matrix(2, 1);
    Matrix *result2 = dgemm(matrix1, vector);
    assert_null(result2);
    free_matrix(&matrix1);
    free_matrix(&vector);

    matrix1 = create_matrix(0, 0);
    Matrix *matrix2 = create_matrix(0, 0);
    Matrix *result3 = dgemm(matrix1, matrix2);
    assert_null(result3);

    free_matrix(&matrix1);
    free_matrix(&matrix2);        
    free_matrix(&m1);
    free_matrix(&m2);
    free_matrix(&result);
    free_matrix(&result2);
    free_matrix(&result3);
}

/******************************************/

/******************************************/

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_add_matrix),
        cmocka_unit_test(test_sub_matrix),
        cmocka_unit_test(test_scale_matrix),
        cmocka_unit_test(test_dotprod),
        cmocka_unit_test(test_compare_matrix),
        cmocka_unit_test(test_dgemm),
    };
    
    return cmocka_run_group_tests(tests, NULL, NULL);
}


