#include <stdint.h>
#include <setjmp.h>
#include <stdarg.h>
#include <stddef.h>
#include <cmocka.h>
#include <stdio.h>
#include <stdlib.h>

//#include "../matrix/matrix.c"
#include "../matrix/matrix.h"

static void test_create_matrix(void **state) {
    Matrix *m = create_matrix(10, 10);
    assert_non_null(m);
    assert_int_equal(m->row, 10);
    assert_int_equal(m->column, 10);

    for (int i = 0; i < 10; i++) {
        assert_non_null(m->value[i]);
    }

    // Libère la mémoire après le test
    for (int i = 0; i < 10; i++) {
        free(m->value[i]);
    }
    free(m->value);
    free(m);

/******************************************/    

    Matrix *m2 = create_matrix(1000, 1000);
    assert_non_null(m2);
    assert_int_equal(m2->row, 1000);
    assert_int_equal(m2->column, 1000);

    for (int i = 0; i < 1000; i++) {
        assert_non_null(m2->value[i]);
    }

    // Libère la mémoire après le test
    for (int i = 0; i < 1000; i++) {
        free(m2->value[i]);
    }
    free(m2->value);
    free(m2);

/******************************************/    

    Matrix *m3 = create_matrix(10000, 10000);
    assert_non_null(m3);
    assert_int_equal(m3->row, 10000);
    assert_int_equal(m3->column, 10000);

    for (int i = 0; i < 10000; i++) {
        assert_non_null(m3->value[i]);
    }

    // Libère la mémoire après le test
    for (int i = 0; i < 10000; i++) {
        free(m3->value[i]);
    }
    free(m3->value);
    free(m3);

/******************************************/    

    assert_null(create_matrix(0, 0));
    assert_null(create_matrix(1, 0));
    assert_null(create_matrix(0, 1));
    assert_null(create_matrix(-1, 1));
    assert_null(create_matrix(1, -1));
    assert_null(create_matrix(-1, -1));
}

/******************************************/

static void test_free_matrix(void **state) {
    Matrix *m = create_matrix(10, 10);
    double vals[100];
    for (int i = 0; i < 100; i++) {
        vals[i] = i;
    }

    fill_matrix(m, vals, 100);

    free_matrix(&m);

    assert_null(m);
}

/******************************************/    

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_create_matrix),
        cmocka_unit_test(test_free_matrix),
    };
    
    return cmocka_run_group_tests(tests, NULL, NULL);
}
