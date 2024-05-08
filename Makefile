# Nom de l'exécutable
TARGET = mlp

# Compiler
CC = gcc

# Flags
CFLAGS = -march=native -Wall -Wextra -g3 -O3 -fopenmp

# Libraries
INCDIRS = -Iinclude

# Paths
SRCS = src/main.c src/networks/feeding.c src/benchmark/bench.c src/networks/mlp_opti.c src/networks/activation.c src/mnist_reader/mnist_reader.c

#
OBJS = $(SRCS:.c=.o)

# Compilation's rule
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS) -lm -lcblas -mavx2 -mfma 
# -I/usr/include/x86_64-linux-gnu -I/opt/intel/oneapi/dal/2023.2.0/include/services/internal/sycl/math/

# Object file
%.o: %.c
	$(CC) $(CFLAGS) $(INCDIRS) -c $< -o $@

# Clean
clean:
	rm -f $(OBJS) $(TARGET)
