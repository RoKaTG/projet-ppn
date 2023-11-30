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

int main(void) {
    const struct CMUnitTest tests[] = {
    };
    
    return cmocka_run_group_tests(tests, NULL, NULL);
}


