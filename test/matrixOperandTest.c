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
    double vals[] = {1.0, 2.0, 3.0, 4.0};
    fill_matrix(m, vals, 4);

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
    scale_matrix(empty, 2.0);
    assert_null(empty->value);

    scale_matrix(NULL, 2.0); 

    free_matrix(&m);
    free_matrix(&empty);
}

/******************************************/    

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_add_matrix),
        cmocka_unit_test(test_sub_matrix),
        cmocka_unit_test(test_scale_matrix),
    };
    
    return cmocka_run_group_tests(tests, NULL, NULL);
}


