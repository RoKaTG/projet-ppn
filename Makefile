CC = gcc
CFLAGS = -O3 -g
LDFLAGS = -lcmocka -lm

SRC_DIR = matrix
TEST_DIR = test
OPERAND_DIR = matrix_operand

SRC_FILES = $(SRC_DIR)/matrix.c $(OPERAND_DIR)/matrixOperand.c
TEST_FILES = $(TEST_DIR)/matrixTest.c
TEST_OPERAND_FILES = $(TEST_DIR)/matrixOperandTest.c
OBJ_FILES = $(SRC_FILES:.c=.o)
TEST_OBJ = $(TEST_FILES:.c=.o)
TEST_OPERAND_OBJ = $(TEST_OPERAND_FILES:.c=.o)

TEST_EXEC = test_matrix
TEST_OPERAND_EXEC = test_operand

.PHONY: all clean test test_operand

all: $(TEST_EXEC) $(TEST_OPERAND_EXEC)

build_test_operand: $(OBJ_FILES) $(TEST_OPERAND_OBJ)
	$(CC) $(CFLAGS) -o $(TEST_OPERAND_EXEC) $^ $(LDFLAGS)

$(TEST_EXEC): $(OBJ_FILES) $(TEST_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

#test: $(TEST_EXEC)
#	./$(TEST_EXEC)

test_operand: build_test_operand
	./$(TEST_OPERAND_EXEC)
	./$(TEST_EXEC)

clean:
	rm -f $(SRC_DIR)/*.o $(TEST_DIR)/*.o $(OPERAND_DIR)/*.o $(TEST_EXEC) $(TEST_OPERAND_EXEC)
