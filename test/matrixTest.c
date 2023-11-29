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

static void test_print_matrix(void **state) {
    Matrix *m = create_matrix(2, 2);
    double vals[] = {1.0, 2.0, 3.0, 4.0};
    fill_matrix(m, vals, 4);

    FILE *temp_file = freopen("temp_output.txt", "w+", stdout);
    if (temp_file == NULL) {
        fail_msg("Failed to redirect stdout to a temp file.");
        return;
    }

    print_matrix(m);

    fflush(stdout);
    fclose(temp_file); 

    FILE *read_file = fopen("temp_output.txt", "r");
    if (read_file == NULL) {
        fail_msg("Failed to open the temporary file for reading.");
        return;
    }

    char buffer[1024] = {0};
    char expected_output[1024];
    sprintf(expected_output, 
            "This is the content of this 2 x 2 matrix:\n%lf \t%lf \t\n%lf \t%lf \t\n", 
            vals[0], vals[1], vals[2], vals[3]);

    size_t num_read = fread(buffer, sizeof(char), 1024, read_file);
    if (num_read == 0 && !feof(read_file)) {
        fail_msg("Failed to read from the temporary file.");
        fclose(read_file);
        return;
    }

    assert_string_equal(buffer, expected_output);

    fclose(read_file);
    remove("temp_output.txt");

    for (int i = 0; i < m->row; i++) {
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

static void test_check_dimensions(void **state) {
    Matrix *m1 = create_matrix(3, 3);
    Size size1 = check_dimensions(m1);
    assert_int_equal(size1.rows, 3);
    assert_int_equal(size1.columns, 3);
    assert_int_equal(size1.type, MATRIX_TYPE);

    Matrix *m2 = create_matrix(3, 1);
    Size size2 = check_dimensions(m2);
    assert_int_equal(size2.rows, 3);
    assert_int_equal(size2.columns, 1);
    assert_int_equal(size2.type, VECTOR_TYPE);

    Matrix *m3 = create_matrix(2, 4);
    Size size3 = check_dimensions(m3);
    assert_int_equal(size3.rows, 2);
    assert_int_equal(size3.columns, 4);
    assert_int_equal(size3.type, MATRIX_TYPE);

    Matrix *m4 = create_matrix(1, 5);
    Size size4 = check_dimensions(m4);
    assert_int_equal(size4.rows, 1);
    assert_int_equal(size4.columns, 5);
    assert_int_equal(size4.type, MATRIX_TYPE);

    Matrix *m5 = NULL;
    Size size5 = check_dimensions(m5);
    assert_int_equal(size5.rows, 0);
    assert_int_equal(size5.columns, 0);
    assert_int_equal(size5.type, MATRIX_TYPE);

    free_matrix(&m1);
    free_matrix(&m2);
    free_matrix(&m3);
    free_matrix(&m4);
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
        cmocka_unit_test(test_print_matrix),
        cmocka_unit_test(test_free_matrix),
        cmocka_unit_test(test_check_dimensions),
        cmocka_unit_test(test_copy_matrix),
        cmocka_unit_test(test_save_matrix),
        cmocka_unit_test(test_load_matrix),
        cmocka_unit_test(test_matrix_randomize),
    };
    
    return cmocka_run_group_tests(tests, NULL, NULL);
}
