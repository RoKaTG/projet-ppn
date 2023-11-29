#include <stdint.h>
#include <setjmp.h>
#include <stdarg.h>
#include <stddef.h>
#include <cmocka.h>
#include <stdio.h>
#include <stdlib.h>

//#include "../matrix/matrix.c"
#include "../matrix/matrix.h"

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

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_copy_matrix),
        cmocka_unit_test(test_save_matrix),
    };
    
    return cmocka_run_group_tests(tests, NULL, NULL);
}
