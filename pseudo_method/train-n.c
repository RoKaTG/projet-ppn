#include "../matrix_operand/matrixOperand.h"
#include "../matrix/matrix.h"

void train(network* net, Matrix* input, Matrix* output, double learning_rate) {
    //Se base sur une struct NN
    //L'entrainement du réseau se découpe en 3 étape :

    // Propagation avant 
        //Calculer les entrées de la couche caché :
        Matrix* Z_hidden=dot(net->hidden_weight,input) ;

        //Calculer les sortie de la couche caché :
        Matrix* A_hidden=apply(sigmoid,Z_hidden);

        //Calculer les entrées de la couche de sortie :
        Matrix* Z_output =dot(net->output_weight,A_hidden);

        //Calculer les sortie de la couche de sortie :
        Matrix* A_output=apply(sigmoid,Z_output);



    // Calcul des Erreurs
        //Calculer les erreurs de la couche de sortie 
        Matrix* E_output=sub(net->output,A_output);

        //Calculer les erreurs de la couche caché :
        Matrix* E_hidden = dot(transpose(net->output_weight),E_output);

    // Retropropagation 
        //Mise à jour des poids de la couche de sortie :
        //Gradient = dérivé de la sigmoid ?
        net->output_weight += scale(learning_rate,dot(mult(E_output,gradient(Z_output)),transpose(A_hidden)));

        //Mise à jour des poids de la couche cachée :
        net->hidden_weight += scale(learning_rate,dot(mult(E_hidden,gradient(Z_hidden)),transpose(input)));
}

