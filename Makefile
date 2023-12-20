CC = gcc
CFLAGS = -O3 -g
LDFLAGS = -lm

SRC_DIR = matrix
OPERAND_DIR = matrix_operand
NN_DIR = neural_network
MNIST_READER_DIR = mnist_reader
TEST_DIR = test

SRC_FILES = $(SRC_DIR)/matrix.c $(OPERAND_DIR)/matrixOperand.c $(NN_DIR)/neuralNetwork.c $(MNIST_READER_DIR)/mnist_reader.c
OBJ_FILES = $(SRC_FILES:.c=.o)

NN_EXEC = $(NN_DIR)/nn_exec
TEST_MATRIX_EXEC = test_matrix
TEST_OPERAND_EXEC = test_operand

TEST_MATRIX_FILES = $(TEST_DIR)/matrixTest.c
TEST_OPERAND_FILES = $(TEST_DIR)/matrixOperandTest.c

.PHONY: mnist build run test_matrix test_operand clean

mnist:
	chmod +x mnist_reader/get-mnist.sh
	./mnist_reader/get-mnist.sh

build: $(NN_EXEC)

$(NN_EXEC): $(OBJ_FILES)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

run: build
	cd $(NN_DIR) && ./nn_exec

test_matrix:
	$(CC) $(CFLAGS) -o $(TEST_MATRIX_EXEC) $(TEST_MATRIX_FILES) $(SRC_FILES) $(LDFLAGS)
	./$(TEST_MATRIX_EXEC)

test_operand:
	$(CC) $(CFLAGS) -o $(TEST_OPERAND_EXEC) $(TEST_OPERAND_FILES) $(SRC_FILES) $(LDFLAGS)
	./$(TEST_OPERAND_EXEC)

clean:
	rm -f $(SRC_DIR)/*.o $(OPERAND_DIR)/*.o $(NN_DIR)/*.o $(MNIST_READER_DIR)/*.o $(NN_EXEC) $(TEST_MATRIX_EXEC) $(TEST_OPERAND_EXEC)
