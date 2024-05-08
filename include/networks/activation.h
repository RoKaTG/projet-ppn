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
/*      Fast Sigmoid's headers        */
/*                                    */
/**************************************/

double fast_sigmoid(double x);
double fast_sigmoidPrime(double x);

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
double reluPrime(double x);

/**************************************/
/*                                    */
/*        leakyreLu's header          */
/*                                    */
/**************************************/

double leakyReLU(double x, double alpha);
double leakyReLUPrime(double x, double alpha);

/**************************************/
/*                                    */
/*          Swish's header            */
/*                                    */
/**************************************/

double swish(double x);
double swishPrime(double x);

/**************************************/
/*                                    */
/*          tanh's header             */
/*                                    */
/**************************************/

double tanhh(double x);
double tanhPrime(double x);

/**************************************/
/*                                    */
/*          AVX2's header             */
/*                                    */
/**************************************/

/******************************reLu******************************/

void relu_avx2(double *x, double *output, int length);
void reluPrime_avx2(double *x, double *output, int length);

/******************************leakyreLu******************************/

void leakyReLU_avx2(double *x, double *output, int length, double alpha);
void leakyReLUPrime_avx2(double *x, double *output, int length, double alpha);

/******************************fastSigmoid******************************/

void fast_sigmoid_avx2(double *x, double *output, int length);
void fast_sigmoidPrime_avx2(double *x, double *output, int length);