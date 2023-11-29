#include <stdint.h>
#include <setjmp.h>
#include <stdarg.h>
#include <stddef.h>
#include <cmocka.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>


//#include "../matrix/matrix.c"
#include "matrix.h"

static void test_create_matrix(void **state) {
    Matrix *m = create_matrix(10, 10);
    assert_non_null(m);
    assert_int_equal(m->row, 10);
    assert_int_equal(m->column, 10);

    for (int i = 0; i < 10; i++) {
        assert_non_null(m->value[i]);
    }

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

static void test_copy_matrix(void **state) {
    Matrix *original = create_matrix(2, 3);
    double vals[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    fill_matrix(original, vals, sizeof(vals) / sizeof(vals[0]));

    Matrix *copy = copy_matrix(original);

    for (int i = 0; i < original->row; i++) {
        for (int j = 0; j < original->column; j++) {
            assert_float_equal(copy->value[i][j], original->value[i][j], 0.0001);
        }
    }

    assert_non_null(copy);
    assert_ptr_not_equal(copy, original);
    assert_ptr_not_equal(copy->value, original->value);

    Matrix *null_copy = copy_matrix(NULL);
    assert_null(null_copy);

    Matrix *empty = create_matrix(0, 0);
    assert_null(empty); 

    Matrix *empty_copy = copy_matrix(empty);
    assert_null(empty_copy); 

    free_matrix(&original);
    free_matrix(&copy);
}

/******************************************/    

static void test_save_matrix(void **state) {
    Matrix *m = create_matrix(2, 2);
    double vals[] = {1.0, 4.0, 3.0, 4.0};
    fill_matrix(m, vals, sizeof(vals) / sizeof(vals[0]));

    assert_int_equal(save_matrix(m, "test_matrix.txt"), 0);

    FILE *file = fopen("test_matrix.txt", "r");
    assert_non_null(file);

    int rows, columns;
    if (fscanf(file, "%d %d\n", &rows, &columns) != 2) {
        fclose(file);
        fail_msg("Failed to read matrix dimensions from file.");
    }
    assert_int_equal(rows, 2);
    assert_int_equal(columns, 2);

    double value;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            if (fscanf(file, "%lf", &value) != 1) {
                fclose(file);
                fail_msg("Failed to read matrix values from file.");
            }
            assert_float_equal(value, m->value[i][j], 0.0001); 
        }
    }

    fclose(file);
  
    free_matrix(&m);
    remove("test_matrix.txt");  
}

/******************************************/    

static void test_load_matrix(void **state) {
    Matrix *original = create_matrix(2, 2);
    double vals[] = {1.0, 2.0, 3.0, 4.0};
    fill_matrix(original, vals, sizeof(vals) / sizeof(vals[0]));

    save_matrix(original, "test_matrix_load.txt");

    Matrix *loaded = load_matrix("test_matrix_load.txt");

    for (int i = 0; i < original->row; i++) {
        for (int j = 0; j < original->column; j++) {
            assert_float_equal(loaded->value[i][j], vals[i * original->column + j], 0.0001);
        }
    }

    free_matrix(&original);
    free_matrix(&loaded);
    remove("test_matrix_load.txt");
}

/******************************************/    

void test_randomize_matrix(void **state) {
}

/******************************************/    

static void test_matrix_randomize(void **state) {
    Matrix *m = create_matrix(2, 2);
    assert_non_null(m);

    double zero_vals[] = {0.0, 0.0, 0.0, 0.0};
    fill_matrix(m, zero_vals, 4);

    matrix_randomize(m, 0.0, 1.0);

    bool changed = false;
    for (int i = 0; i < m->row; i++) {
        for (int j = 0; j < m->column; j++) {
            if (m->value[i][j] != 0.0) {
                changed = true;
                break;
            }
        }
    }
    assert_true(changed);

    free_matrix(&m);
}

/******************************************/    

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_create_matrix),
        cmocka_unit_test(test_fill_matrix),
        cmocka_unit_test(test_free_matrix),
        cmocka_unit_test(test_copy_matrix),
        cmocka_unit_test(test_save_matrix),
        cmocka_unit_test(test_load_matrix),
        cmocka_unit_test(test_matrix_randomize),
    };
    
    return cmocka_run_group_tests(tests, NULL, NULL);
}
