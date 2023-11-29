#include <stdint.h>
#include <setjmp.h>
#include <stdarg.h>
#include <stddef.h>
#include <cmocka.h>
#include <stdio.h>
#include <stdlib.h>

//#include "../matrix/matrix.c"
#include "../matrix/matrix.h"

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

static void test_check_dimensions(void **state) {
    // Test avec une matrice standard
    Matrix *m1 = create_matrix(3, 3);
    Size size1 = check_dimensions(m1);
    assert_int_equal(size1.rows, 3);
    assert_int_equal(size1.columns, 3);
    assert_int_equal(size1.type, MATRIX_TYPE);

    // Test avec un vecteur (une seule colonne)
    Matrix *m2 = create_matrix(3, 1);
    Size size2 = check_dimensions(m2);
    assert_int_equal(size2.rows, 3);
    assert_int_equal(size2.columns, 1);
    assert_int_equal(size2.type, VECTOR_TYPE);

    // Test avec une autre matrice standard
    Matrix *m3 = create_matrix(2, 4);
    Size size3 = check_dimensions(m3);
    assert_int_equal(size3.rows, 2);
    assert_int_equal(size3.columns, 4);
    assert_int_equal(size3.type, MATRIX_TYPE);

    // Test avec un vecteur (ligne unique)
    Matrix *m4 = create_matrix(1, 5);
    Size size4 = check_dimensions(m4);
    assert_int_equal(size4.rows, 1);
    assert_int_equal(size4.columns, 5);
    assert_int_equal(size4.type, MATRIX_TYPE);  // Ligne unique est considérée comme une matrice

    // Test avec une matrice vide ou NULL
    Matrix *m5 = NULL;
    Size size5 = check_dimensions(m5);
    assert_int_equal(size5.rows, 0);
    assert_int_equal(size5.columns, 0);
    assert_int_equal(size5.type, MATRIX_TYPE);

    // Nettoyage
    free_matrix(&m1);
    free_matrix(&m2);
    free_matrix(&m3);
    free_matrix(&m4);
    // Pas besoin de libérer m5 car il est NULL
}

/******************************************/

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_print_matrix),
        cmocka_unit_test(test_check_dimensions),
    };
    
    return cmocka_run_group_tests(tests, NULL, NULL);
}
