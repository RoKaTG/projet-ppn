#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <stdbool.h>

#include "neuralNetwork.h"
#include "../mnist_reader/mnist_reader.h"
#include "../matrix_operand/matrixOperand.h"

// Fonctions d'activation et leurs dérivées
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double x) {
    double sigmoid_x = sigmoid(x);
    return sigmoid_x * (1 - sigmoid_x);
}


/*void apply_function(Matrix* m, double (*func)(double)) {
    for (int i = 0; i < m->row; i++) {
        for (int j = 0; j < m->column; j++) {
            m->value[i][j] = func(m->value[i][j]);
        }
    }
}*/

void apply_function(Matrix* m, double (*func)(double)) {
    printf("Applying function at address %p\n", (void*)m);
    for (int i = 0; i < m->row; i++) {
        for (int j = 0; j < m->column; j++) {
            m->value[i][j] = func(m->value[i][j]);
        }
    }
}

// Fonction pour appliquer la dérivée d'une fonction d'activation à chaque élément d'une matrice
void apply_function_derivative(Matrix* m, double (*func)(double)) {
    for (int i = 0; i < m->row; i++) {
        for (int j = 0; j < m->column; j++) {
            m->value[i][j] = func(m->value[i][j]);
        }
    }
}

/*void softmax(Matrix* m) {
    double sum = 0.0;
    for (int i = 0; i < m->row; i++) {
        m->value[i][0] = exp(m->value[i][0]);
        sum += m->value[i][0];
    }
    for (int i = 0; i < m->row; i++) {
        m->value[i][0] /= sum;
    }
}*/

void softmax(Matrix* m) {
    if (m == NULL) {
        printf("Error: softmax received a NULL matrix pointer\n");
        return;
    }
    printf("Calling softmax on matrix at address %p\n", (void*)m);
    double sum = 0.0;
    for (int i = 0; i < m->row; i++) {
        m->value[i][0] = exp(m->value[i][0]);
        sum += m->value[i][0];
    }
    for (int i = 0; i < m->row; i++) {
        m->value[i][0] /= sum;
    }
}

// Fonction pour calculer la dérivée de la fonction softmax
void softmax_derivative(Matrix* m, Matrix* expected_output) {
    for (int i = 0; i < m->row; i++) {
        for (int j = 0; j < m->column; j++) {
            if (expected_output->value[i][j] == 1.0) {
                m->value[i][j] = m->value[i][j] - 1.0;
            }
        }
    }
}

/*********************************************************************************/

// Fonction pour initialiser les couches du réseau
/*NeuralLayer* initialize_network(int number_of_layers, const int layer_sizes[]) {
    NeuralLayer* layers = (NeuralLayer*) malloc(number_of_layers * sizeof(NeuralLayer));
    if (layers == NULL) {
        fprintf(stderr, "Erreur d'allocation de mémoire pour les couches du réseau.\n");
        return NULL;
    }

    for (int i = 0; i < number_of_layers; i++) {
        layers[i].weights = create_matrix(layer_sizes[i + 1], layer_sizes[i]);
        layers[i].biases = create_matrix(layer_sizes[i + 1], 1);
        layers[i].activation_func = (i == number_of_layers - 1) ? softmax : sigmoid;
        matrix_randomize(layers[i].weights, 0.0, 1.0);
        matrix_randomize(layers[i].biases, 0.0, 1.0);
    }
    return layers;
}*/

/*NeuralLayer* initialize_network(int number_of_layers, const int layer_sizes[]) {
    NeuralLayer* layers = (NeuralLayer*) malloc(number_of_layers * sizeof(NeuralLayer));
    if (layers == NULL) {
        fprintf(stderr, "Erreur d'allocation de mémoire pour les couches du réseau.\n");
        return NULL;
    }

    for (int i = 0; i < number_of_layers; i++) {
        printf("Initializing layer %d: size %d x %d\n", i, layer_sizes[i + 1], layer_sizes[i]);
        layers[i].weights = create_matrix(layer_sizes[i + 1], layer_sizes[i]);
        layers[i].biases = create_matrix(layer_sizes[i + 1], 1);
        layers[i].activation_func = (i == number_of_layers - 1) ? softmax : sigmoid;
        matrix_randomize(layers[i].weights, 0.0, 1.0);
        matrix_randomize(layers[i].biases, 0.0, 1.0);
    }
    return layers;
}*/

/*NeuralLayer* initialize_network(int number_of_layers, const int layer_sizes[]) {
    NeuralLayer* layers = (NeuralLayer*) malloc(number_of_layers * sizeof(NeuralLayer));
    if (layers == NULL) {
        fprintf(stderr, "Erreur d'allocation de mémoire pour les couches du réseau.\n");
        return NULL;
    }

    for (int i = 0; i < number_of_layers; i++) {
        printf("Initializing layer %d: size %d x %d\n", i, layer_sizes[i + 1], layer_sizes[i]);
        layers[i].weights = create_matrix(layer_sizes[i + 1], layer_sizes[i]);
        layers[i].biases = create_matrix(layer_sizes[i + 1], 1);
        layers[i].activation_func = (i == number_of_layers - 1) ? softmax : sigmoid;
        matrix_randomize(layers[i].weights, 0.0, 1.0);
        matrix_randomize(layers[i].biases, 0.0, 1.0);
    }
    return layers;
}*/

NeuralLayer* initialize_network(int number_of_layers, const int layer_sizes[]) {
    NeuralLayer* layers = (NeuralLayer*) malloc(number_of_layers * sizeof(NeuralLayer));
    if (layers == NULL) {
        fprintf(stderr, "Erreur d'allocation de mémoire pour les couches du réseau.\n");
        return NULL;
    }

    for (int i = 0; i < number_of_layers; i++) {
        int layer_size = (i == number_of_layers - 1) ? layer_sizes[i] : layer_sizes[i + 1];
        printf("Initializing layer %d: size %d x %d\n", i, layer_size, layer_sizes[i]);
        layers[i].weights = create_matrix(layer_size, layer_sizes[i]);
        layers[i].biases = create_matrix(layer_size, 1);
        layers[i].activation_func = (i == number_of_layers - 1) ? softmax : sigmoid;
        matrix_randomize(layers[i].weights, 0.0, 1.0);
        matrix_randomize(layers[i].biases, 0.0, 1.0);
    }
    return layers;
}



/****************************************************************************************/

/*Matrix* forward_propagation(NeuralLayer* network, int number_of_layers, Matrix* input) {
    Matrix* current_input = copy_matrix(input);
    for (int i = 0; i < number_of_layers; i++) {
        NeuralLayer layer = network[i];
        Matrix* z = dgemm(layer.weights, current_input); // z = weights * input
        add_matrix(z, layer.biases); // z = z + biases
        apply_function(z, layer.activation_func); // Activation function

        free_matrix(&current_input); // Free previous input
        current_input = z; // Set current input to output of this layer
    }
    return current_input; // Output of last layer is the output of the network
}*/

/*Matrix* forward_propagation(NeuralLayer* network, int number_of_layers, Matrix* input) {
    Matrix* current_input = copy_matrix(input);
    Matrix* next_input;

    for (int i = 0; i < number_of_layers; i++) {
        NeuralLayer layer = network[i];
        next_input = dgemm(layer.weights, current_input); // Calcule le produit matriciel
        add_matrix(next_input, layer.biases); // Ajoute les biais
        apply_function(next_input, layer.activation_func); // Applique la fonction d'activation

        if (i > 0) {
            free_matrix(&current_input); // Libère la mémoire de l'input précédent
        }
        current_input = next_input; // Réassigne current_input à next_input pour la prochaine itération
    }
    return current_input; // Retourne le résultat de la dernière couche
}*/

Matrix* forward_propagation(NeuralLayer* network, int number_of_layers, Matrix* input) {
    Matrix* current_input = copy_matrix(input);
    Matrix* next_input;

    for (int i = 0; i < number_of_layers; i++) {
        printf("Layer %d: input address %p\n", i, (void*)current_input);
        next_input = dgemm(network[i].weights, current_input);
        add_matrix(next_input, network[i].biases);
        apply_function(next_input, network[i].activation_func);

        if (i > 0) {
            free_matrix(&current_input);
        }
        current_input = next_input;
    }
    return current_input;
}


void backpropagation(NeuralLayer* network, int number_of_layers, Matrix* input, Matrix* expected_output, double learning_rate) {
    // Étape 1 : Calcul de la sortie du réseau (forward propagation)
    Matrix** activations = (Matrix**)malloc(number_of_layers * sizeof(Matrix*));
    Matrix* current_input = copy_matrix(input);
    activations[0] = current_input; // La première activation est l'entrée

    for (int i = 0; i < number_of_layers; i++) {
        current_input = dgemm(network[i].weights, current_input); // z = weights * input
        add_matrix(current_input, network[i].biases); // z = z + biases
        apply_function(current_input, network[i].activation_func); // Activation function
        activations[i + 1] = current_input; // Stocker l'activation pour la rétropropagation
    }

    // Étape 2 : Calcul de l'erreur de sortie
    Matrix* error = sub_matrix(expected_output, activations[number_of_layers]);

    // Appliquer la dérivée de softmax si la couche de sortie utilise softmax
    if (network[number_of_layers - 1].activation_func == softmax) {
        softmax_derivative(error, expected_output);
    }

    // Étape 3 : Propagation de l'erreur en arrière à travers le réseau
    for (int i = number_of_layers - 1; i >= 0; i--) {
        Matrix* delta;
        if (network[i].activation_func == softmax) {
            // Pour softmax, l'erreur a déjà été calculée ci-dessus
            delta = copy_matrix(error);
        } else {
            // Appliquer la dérivée de la fonction d'activation
            delta = copy_matrix(error);
            apply_function_derivative(delta, sigmoid_derivative);
        }

        // Calcul du gradient par rapport aux poids
        Matrix* previous_activation_transposed = transpose_matrix(activations[i]);
        Matrix* weight_gradient = dgemm(delta, previous_activation_transposed);

        // Mise à jour des poids
        scale_matrix(weight_gradient, learning_rate);
        add_matrix(network[i].weights, weight_gradient);

        // Mise à jour des biais
        scale_matrix(delta, learning_rate);
        add_matrix(network[i].biases, delta);

        // Calcul de l'erreur pour la couche précédente (si i > 0)
        if (i > 0) {
            Matrix* weights_transposed = transpose_matrix(network[i].weights);
            free_matrix(&error);
            error = dgemm(weights_transposed, delta);
            free_matrix(&weights_transposed);
        }

        free_matrix(&delta);
        free_matrix(&previous_activation_transposed);
        free_matrix(&weight_gradient);
    }

    // Libération des ressources
    for (int i = 0; i <= number_of_layers; i++) {
        free_matrix(&activations[i]);
    }
    free(activations);
    free_matrix(&error);
}

/***********************************************************************/

int main() {
    srand((unsigned int)time(NULL));

    const int number_of_layers = 3;
    const int layer_sizes[] = {784, 128, 10}; // Tailles des couches: entrée, cachée, sortie
    NeuralLayer* network = initialize_network(number_of_layers, layer_sizes);

    const int epochs = 10; // Nombre d'itérations sur l'ensemble des données
    const int batch_size = 10; // Nombre d'exemples dans chaque batch
    const double learning_rate = 0.01;
    const int input_size = 784; // 28x28 images
    const int output_size = 10; // 10 classes pour les chiffres de 0 à 9
    const int total_images = 1000; // Nombre total d'images à utiliser pour l'entraînement

    FILE* imageFile = fopen("../mnist/train-images-idx3-ubyte", "rb");
    FILE* labelFile = fopen("../mnist/train-labels-idx1-ubyte", "rb");

    for (int epoch = 0; epoch < epochs; epoch++) {
        for (int batch = 0; batch < total_images / batch_size; batch++) {
            uint8_t* images = readMnistImages(imageFile, batch * batch_size, batch_size);
            uint8_t* labels = readMnistLabels(labelFile, batch * batch_size, batch_size);

            for (int i = 0; i < batch_size; i++) {
                Matrix* input = create_matrix(input_size, 1);
                for (int j = 0; j < input_size; j++) {
                    input->value[j][0] = images[i * input_size + j] / 255.0;
                }

                Matrix* expected_output = create_matrix(output_size, 1);
                for (int j = 0; j < output_size; j++) {
                    expected_output->value[j][0] = (j == labels[i]) ? 1.0 : 0.0;
                }

                Matrix* output = forward_propagation(network, number_of_layers, input);
                backpropagation(network, number_of_layers, input, expected_output, learning_rate);

                free_matrix(&input);
                free_matrix(&expected_output);
                free_matrix(&output);
            }

            free(images);
            free(labels);
        }

        printf("Epoch %d complete\n", epoch);
    }

    fclose(imageFile);
    fclose(labelFile);

    for (int i = 0; i < number_of_layers; i++) {
        free_matrix(&network[i].weights);
        free_matrix(&network[i].biases);
    }
    free(network);

    return 0;
}


