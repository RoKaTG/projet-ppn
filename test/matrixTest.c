#include <stdint.h>
#include <setjmp.h>
#include <stdarg.h>
#include <stddef.h>
#include <cmocka.h>
#include <stdio.h>
#include <stdlib.h>

//#include "../matrix/matrix.c"
#include "../matrix/matrix.h"

static void test_fill_matrix(void **state) {
    int rows = 2;
    int columns = 3;
    double vals_full[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    int vals_full_size = sizeof(vals_full) / sizeof(vals_full[0]);

    Matrix *m = create_matrix(rows, columns);
    assert_non_null(m);
    int res = fill_matrix(m, vals_full, vals_full_size);
    assert_int_equal(res, 0);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            assert_float_equal(m->value[i][j], vals_full[i * columns + j], 0.0001);
        }
    }
    for (int i = 0; i < rows; i++) {
        free(m->value[i]);
    }
    free(m->value);
    free(m);

    m = create_matrix(rows, columns);
    assert_non_null(m);

    double vals_partial[] = {1.0, 2.0, 3.0};
    int vals_partial_size = sizeof(vals_partial) / sizeof(vals_partial[0]);
    res = fill_matrix(m, vals_partial, vals_partial_size);
    assert_int_equal(res, -1);

    for (int i = 0; i < rows; i++) {
        free(m->value[i]);
    }
    free(m->value);
    free(m);
}

/******************************************/    

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_fill_matrix),
    };
    
    return cmocka_run_group_tests(tests, NULL, NULL);
}
