#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "neuralNetwork.h"
#include "../mnist_reader/mnist_reader.h"
#include "../matrix_operand/matrixOperand.h"

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

void apply_function(Matrix* m, double (*func)(double)) {
    for (int i = 0; i < m->row; i++) {
        for (int j = 0; j < m->column; j++) {
            m->value[i][j] = func(m->value[i][j]);
        }
    }
}

void softmax(Matrix* m) {
    double sum = 0.0;
    for (int i = 0; i < m->row; i++) {
        m->value[i][0] = exp(m->value[i][0]);
        sum += m->value[i][0];
    }
    for (int i = 0; i < m->row; i++) {
        m->value[i][0] /= sum;
    }
}

void backpropagate(Matrix* input, Matrix* output, Matrix* expected, Matrix* weights, Matrix* biases, double learning_rate) {
    double error = expected->value[0][0] - output->value[0][0];
    double derivative = output->value[0][0] * (1 - output->value[0][0]);
    double gradient = error * derivative;
    
    for (int i = 0; i < weights->row; i++) {
        for (int j = 0; j < weights->column; j++) {
            weights->value[i][j] += learning_rate * gradient * input->value[j][0];
        }
        biases->value[i][0] += learning_rate * gradient;
    }
}

int main() {
    srand((unsigned int)time(NULL));

    const int input_size = 784;
    const int hidden_size = 128;
    const int output_size = 10;
    const double learning_rate = 0.01;
    const int epochs = 1000;
    const int number_of_images = 10;

    Matrix* input_layer = create_matrix(input_size, 1);
    Matrix* hidden_layer_weights = create_matrix(hidden_size, input_size);
    Matrix* hidden_layer_biases = create_matrix(hidden_size, 1);
    Matrix* output_layer_weights = create_matrix(output_size, hidden_size);
    Matrix* output_layer_biases = create_matrix(output_size, 1);
    Matrix* expected_output = create_matrix(output_size, 1);

    matrix_randomize(hidden_layer_weights, 0.0, 1.0);
    matrix_randomize(hidden_layer_biases, 0.0, 1.0);
    matrix_randomize(output_layer_weights, 0.0, 1.0);
    matrix_randomize(output_layer_biases, 0.0, 1.0);

    FILE* imageFile = fopen("../mnist/train-images-idx3-ubyte", "rb");
    FILE* labelFile = fopen("../mnist/train-labels-idx1-ubyte", "rb");
    uint8_t* images = readMnistImages(imageFile, 0, number_of_images);
    uint8_t* labels = readMnistLabels(labelFile, 0, number_of_images);

    for (int epoch = 0; epoch < epochs; epoch++) {
        for (int i = 0; i < number_of_images; i++) {
            for (int j = 0; j < input_size; j++) {
                input_layer->value[j][0] = images[i * input_size + j] / 255.0;
            }

            for (int j = 0; j < output_size; j++) {
                expected_output->value[j][0] = (labels[i] == j) ? 1.0 : 0.0;
            }
            
            Matrix* hidden_layer_input = dgemm(hidden_layer_weights, input_layer);
            add_matrix(hidden_layer_input, hidden_layer_biases);
            apply_function(hidden_layer_input, sigmoid);

            Matrix* output_layer_input = dgemm(output_layer_weights, hidden_layer_input);
            add_matrix(output_layer_input, output_layer_biases);
            softmax(output_layer_input);
            
            backpropagate(hidden_layer_input, output_layer_input, expected_output, output_layer_weights, output_layer_biases, learning_rate);
            
            if (epoch % 100 == 0) {
                printf("Epoch %d: Image %d, Label: %d, Output: [", epoch, i, labels[i]);
                for (int j = 0; j < output_size; j++) {
                    printf("%.2f ", output_layer_input->value[j][0]);
                }
                printf("]\n");
            }

            free_matrix(&hidden_layer_input);
            free_matrix(&output_layer_input);
        }
    }
    
    free(images);
    free(labels);
    fclose(imageFile);
    fclose(labelFile);

    free_matrix(&input_layer);
    free_matrix(&hidden_layer_weights);
    free_matrix(&hidden_layer_biases);
    free_matrix(&output_layer_weights);
    free_matrix(&output_layer_biases);
    free_matrix(&expected_output);

    return 0;
}
