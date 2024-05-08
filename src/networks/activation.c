#include <cblas.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <stdint.h>
#include <immintrin.h>
#include <omp.h>


#include "../../include/networks/activation.h"

/**************************************/
/*                                    */
/*          Sigmoid's Function        */
/*                                    */
/**************************************/

/**
 * Sigmoid activation function.
 * Computes the sigmoid output of an input value.
 *
 * @param x The input value.
 * @return The output value of the sigmoid function.
 */
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

/**
 * Derivative of the sigmoid activation function.
 * Used in computing the gradient during backpropagation.
 *
 * @param x The input value.
 * @return The derivative of the sigmoid function at that point.
 */
double sigmoidPrime(double x) {    
    double sig = sigmoid(x);
    return sig * (1.0 - sig);
}

/**************************************/
/*                                    */
/*      Fast Sigmoid's Function       */
/*                                    */
/**************************************/

/**
 * Fast sigmoid function using double precision.
 * An approximation of the sigmoid function that is computationally faster.
 * It is defined as f(x) = x / (1 + |x|).
 *
 * @param x The input value (double).
 * @return The fast sigmoid of x (double).
 */
double fast_sigmoid(double x) {
    return x / (1.0 + fabs(x));
}

/**
 * Derivative of the fast sigmoid function using double precision.
 * Provides the gradient of the fast sigmoid function, useful in optimization algorithms like backpropagation.
 * For f(x) = x / (1 + |x|), the derivative is f'(x) = 1 / ((1 + |x|)²).
 *
 * @param x The input value (double).
 * @return The derivative of the fast sigmoid at x (double).
 */
double fast_sigmoidPrime(double x) {
    double abs_x = fabs(x);
    return 1.0 / ((1 + abs_x) * (1 + abs_x));
}

/**************************************/
/*                                    */
/*          Softmax's Function        */
/*                                    */
/**************************************/

/**
 * Apply the softmax function to the logits to compute probabilities.
 * Softmax function normalizes the logits and converts them into probabilities.
 * This prevents numerical stability issues.
 *
 * @param logits The input logits array.
 * @param probabilities The output probabilities array.
 * @param length The length of the logits and probabilities arrays.
 */
void softmax(double *logits, double *probabilities, int length) {
    double max_logit = -INFINITY; // Search for the maximum logit
    for (int i = 0; i < length; i++) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
        }
    }

    double sum = 0.0;
    
    for (int i = 0; i < length; i++) {
        probabilities[i] = exp(logits[i] - max_logit); // Prevents numerical stability issues
        sum += probabilities[i];
    }
    
    for (int i = 0; i < length; i++) {
        probabilities[i] /= sum;
    }
}

/**************************************/
/*                                    */
/*          reLu's Function           */
/*                                    */
/**************************************/

/**
 * Rectified Linear Unit (ReLU) activation function.
 *
 * @param x Input value.
 * @return Output value after applying ReLU activation.
 */
double relu(double x) {
    return (x > 0) ? x : 0;
}

/**
 * Derivative of the Rectified Linear Unit (ReLU) activation function.
 *
 * @param x Input value.
 * @return Derivative of the ReLU activation function at the given input.
 */
double reluPrime(double x) {
    return (x > 0) ? 1 : 0;
}

/**************************************/
/*                                    */
/*       LeakyreLu's Function         */
/*                                    */
/**************************************/

/**
 * Leaky ReLU activation function.
 * Applies the Leaky ReLU activation to a double-precision floating-point number.
 *
 * @param x The input value.
 * @param alpha The leakage factor for negative inputs.
 * @return The Leaky ReLU activation of x.
 */
double leakyReLU(double x, double alpha) {
    return x > 0 ? x : alpha * x;
}

/**
 * Derivative of the Leaky ReLU activation function.
 * Computes the derivative of the Leaky ReLU activation for a double-precision floating-point number.
 *
 * @param x The input value.
 * @param alpha The leakage factor for negative inputs.
 * @return The derivative of the Leaky ReLU activation at x.
 */
double leakyReLUPrime(double x, double alpha) {
    return x > 0 ? 1.0 : alpha;
}

/**************************************/
/*                                    */
/*          Swish's Function          */
/*                                    */
/**************************************/

/**
 * Swish activation function.
 * Computes the Swish activation of a given double-precision floating-point number.
 * Swish is defined as f(x) = x / (1 + e^{-x}).
 *
 * @param x The input value.
 * @return The Swish activation of x.
 */
double swish(double x) {
    return x / (1.0 + exp(-x));
}

/**
 * Derivative of the Swish activation function.
 * Computes the derivative of the Swish activation for a given double-precision floating-point number.
 * The derivative is f'(x) = f(x) + sigmoid(x) * (1 - f(x)), where f(x) is the Swish function.
 *
 * @param x The input value.
 * @return The derivative of the Swish activation at x.
 */
double swishPrime(double x) {
    double swish_x = swish(x);
    double sigmoid_x = 1.0 / (1.0 + exp(-x));
    return swish_x + sigmoid_x * (1 - swish_x);
}

/**************************************/
/*                                    */
/*          tanh's Function           */
/*                                    */
/**************************************/

/**
 * Hyperbolic tangent function.
 * Computes the hyperbolic tangent of a given double-precision floating-point number.
 *
 * @param x The input value.
 * @return The hyperbolic tangent of x.
 */
double tanhh(double x) {
    return tanh(x);
}

/**
 * Derivative of the hyperbolic tangent function.
 * Computes the derivative of the hyperbolic tangent for a given double-precision floating-point number.
 *
 * @param x The input value.
 * @return The derivative of the hyperbolic tangent at x.
 */
double tanhPrime(double x) {
    double tanh_x = tanh(x);
    return 1.0 - tanh_x * tanh_x;
}

/**************************************/
/*                                    */
/*          AVX2 counterpart          */
/*                                    */
/**************************************/

/******************************reLU******************************/

/**
 * ReLU activation function using AVX2 intrinsics.
 * Applies the Rectified Linear Unit (ReLU) activation to an array of double-precision floating-point numbers.
 * For each element x, it computes max(x, 0).
 *
 * @param x Pointer to the input array (double precision).
 * @param output Pointer to the output array where results are stored.
 * @param length The number of elements in the input and output arrays.
 */
void relu_avx2(double *x, double *output, int length) {
    __m256d zero = _mm256_setzero_pd();
    for (int i = 0; i < length; i += 8) {
        __m256d x_vec = _mm256_loadu_pd(&x[i]);
        __m256d result = _mm256_max_pd(x_vec, zero);
        _mm256_storeu_pd(&output[i], result);
    }
}

/**
 * Derivative of the ReLU activation function using AVX2 intrinsics.
 * Computes the derivative of the Rectified Linear Unit (ReLU) activation for an array of double-precision floating-point numbers.
 * For each element x, it returns 1 if x > 0, otherwise 0.
 *
 * @param x Pointer to the input array (double precision).
 * @param output Pointer to the output array where results are stored.
 * @param length The number of elements in the input and output arrays.
 */
void reluPrime_avx2(double *x, double *output, int length) {
    __m256d zero = _mm256_setzero_pd();
    for (int i = 0; i < length; i += 8) {
        __m256d x_vec = _mm256_loadu_pd(&x[i]);
        __m256d mask = _mm256_cmp_pd(x_vec, zero, _CMP_GT_OS);
        __m256d relu_prime = _mm256_and_pd(mask, _mm256_set1_pd(1.0));

        _mm256_storeu_pd(&output[i], relu_prime);
    }
}

/******************************LeakyreLu******************************/

/**
 * Leaky ReLU activation function using AVX2 intrinsics.
 * Applies the Leaky ReLU activation to an array of double-precision floating-point numbers.
 * For each element x, it computes x if x > 0, otherwise alpha * x.
 *
 * @param x Pointer to the input array (double precision).
 * @param output Pointer to the output array where results are stored.
 * @param length The number of elements in the input and output arrays.
 * @param alpha The leakage factor for negative inputs.
 */
void leakyReLU_avx2(double *x, double *output, int length, double alpha) {
    __m256d zero = _mm256_setzero_pd();
    __m256d alpha_vec = _mm256_set1_pd(alpha);
    
    for (int i = 0; i < length; i += 4) {
        __m256d x_vec = _mm256_loadu_pd(&x[i]);
        __m256d neg_mask = _mm256_cmp_pd(x_vec, zero, _CMP_LT_OS);
        __m256d alpha_mul = _mm256_mul_pd(alpha_vec, x_vec);
        __m256d result = _mm256_blendv_pd(x_vec, alpha_mul, neg_mask);
        _mm256_storeu_pd(&output[i], result);
    }
}

/**
 * Derivative of the Leaky ReLU activation function using AVX2 intrinsics.
 * Computes the derivative of the Leaky ReLU activation for an array of double-precision floating-point numbers.
 * For each element x, it returns 1 if x > 0, otherwise alpha.
 *
 * @param x Pointer to the input array (double precision).
 * @param output Pointer to the output array where results are stored.
 * @param length The number of elements in the input and output arrays.
 * @param alpha The leakage factor for negative inputs.
 */
void leakyReLUPrime_avx2(double *x, double *output, int length, double alpha) {
    __m256d zero = _mm256_setzero_pd();
    __m256d alpha_vec = _mm256_set1_pd(alpha);
    __m256d one = _mm256_set1_pd(1.0);

    for (int i = 0; i < length; i += 4) {
        __m256d x_vec = _mm256_loadu_pd(&x[i]);
        __m256d neg_mask = _mm256_cmp_pd(x_vec, zero, _CMP_LT_OS);
        __m256d result = _mm256_blendv_pd(one, alpha_vec, neg_mask);
        _mm256_storeu_pd(&output[i], result);
    }
}

/******************************fastSigmoid******************************/

/**
 * Fast sigmoid function using AVX2 intrinsics with double precision.
 * This function applies the fast sigmoid to each element of the input array x,
 * and stores the result in the output array.
 * The arrays should be allocated with a size that is a multiple of 4.
 *
 * @param x Pointer to the input array (double precision).
 * @param output Pointer to the output array where results are stored.
 * @param length The number of elements in the input and output arrays.
 */
void fast_sigmoid_avx2(double *x, double *output, int length) {
    __m256d one = _mm256_set1_pd(1.0);
    for (int i = 0; i < length; i += 4) {
        __m256d x_vec = _mm256_loadu_pd(&x[i]);
        __m256d abs_x = _mm256_andnot_pd(_mm256_set1_pd(-0.0), x_vec); // Absolute value of x
        __m256d result = _mm256_div_pd(x_vec, _mm256_add_pd(one, abs_x));
        _mm256_storeu_pd(&output[i], result);
    }
}

/**
 * Derivative of the fast sigmoid function using AVX2 intrinsics with double precision.
 * This function computes the derivative of the fast sigmoid for each element of the input array x,
 * and stores the result in the output array.
 * The arrays should be allocated with a size that is a multiple of 4.
 *
 * @param x Pointer to the input array (double precision).
 * @param output Pointer to the output array where results are stored.
 * @param length The number of elements in the input and output arrays.
 */
void fast_sigmoidPrime_avx2(double *x, double *output, int length) {
    __m256d one = _mm256_set1_pd(1.0);
    for (int i = 0; i < length; i += 4) {
        __m256d x_vec = _mm256_loadu_pd(&x[i]);
        __m256d abs_x = _mm256_andnot_pd(_mm256_set1_pd(-0.0), x_vec); // Absolute value of x
        __m256d denom = _mm256_mul_pd(_mm256_add_pd(one, abs_x), _mm256_add_pd(one, abs_x));
        __m256d result = _mm256_div_pd(one, denom);
        _mm256_storeu_pd(&output[i], result);
    }
}