void test_neural_network(NeuralNetwork* nn, Image** test_data) {
    double total_loss = 0.0;
    int correct_predictions = 0;

    //utiliser 20 images de 100 (20% du dataset).
    for (int i=80;i<100;i++) {
        // Propagation avant
        Matrix* input_matrix = test_data[i]->pixels;
        Matrix* predicted = forward_propagation(nn, input_matrix);

        // Calculer la perte
        double loss = calculate_loss(predicted, target_matrix);

        // Accumuler la perte pour l'évaluation moyenne
        total_loss += loss;

        // Comparer la prédiction avec le label réel
        int predicted_label= /transformer la sortie prédite en label/;
        int actual_label=test_data[i]->label;
        if (predicted_label==actual_label) {
            correct_predictions++;
        }
    }

    // Calculer la perte moyenne sur l'ensemble de test
    double average_loss = total_loss/20;

    // Calculer la précision (accuracy)
    double accuracy = (double)correct_predictions/20;

    // Afficher les résultats
    printf("Test Loss: %lf\n", average_loss);
    printf("Accuracy: %.2lf%%\n", accuracy*100);
}
