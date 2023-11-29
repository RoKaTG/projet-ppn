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

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_print_matrix),
    };
    
    return cmocka_run_group_tests(tests, NULL, NULL);
}
