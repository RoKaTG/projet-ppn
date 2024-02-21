#pragma once

/**************************************/
/*                                    */
/*          Sigmoid's headers         */
/*                                    */
/**************************************/

double sigmoid(double x);
double sigmoidPrime(double x);

/**************************************/
/*                                    */
/*          Softmax's header          */
/*                                    */
/**************************************/

void softmax(double *logits, double *probabilities, int length);

/**************************************/
/*                                    */
/*          reLu's header             */
/*                                    */
/**************************************/

double relu(double x);