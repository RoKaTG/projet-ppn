CC = gcc
CFLAGS = -O3 -g
LDFLAGS = -lcmocka -lm

SRC_DIR = matrix
TEST_DIR = test

SRC_FILES = $(SRC_DIR)/matrix.c
TEST_FILES = $(TEST_DIR)/matrixTest.c
OBJ_FILES = $(SRC_FILES:.c=.o) $(TEST_FILES:.c=.o)

TEST_EXEC = test_matrix

.PHONY: all clean test

all: $(TEST_EXEC)

$(TEST_EXEC): $(OBJ_FILES)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

test: $(TEST_EXEC)
	./$(TEST_EXEC)

clean:
	rm -f $(SRC_DIR)/*.o $(TEST_DIR)/*.o $(TEST_EXEC)
