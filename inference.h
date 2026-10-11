#ifndef INFERENCE_H
#define INFERENCE_H
#include <ap_fixed.h>
#include <cstdint>
#include <hls_stream.h>
#include <hls_math.h>
#include <complex>

// Configuration constants
#define IMAGE_HEIGHT 64
#define IMAGE_WIDTH 64
#define IN_CHANNELS 1
#define NUM_CLASSES 94
#define NUM_LAYERS 3

// Layer configurations
#define LAYER1_KERNELS 2
#define LAYER2_KERNELS 4
#define LAYER3_KERNELS 16

// Maximum dimensions for buffer allocation
#define MAX_HEIGHT 64
#define MAX_WIDTH 64
#define MAX_CHANNELS 16

// Single uniform precision throughout - using ap_fixed<24,14>
// Both images and all layer weights/biases/bn params use this type.
typedef ap_fixed<24, 14> fixed_t;
typedef std::complex<fixed_t> complex_fixed_t;

// FFT Constants
#define PI 3.14159265358979323846

// Helper macro for conversion
#define FIXED_TO_FLOAT(x) ((float)(x))

// Function declarations
void fft_1d(complex_fixed_t input[], complex_fixed_t output[], int N, bool inverse);
void fft_2d(complex_fixed_t input[IMAGE_HEIGHT][IMAGE_WIDTH],
            complex_fixed_t output[IMAGE_HEIGHT][IMAGE_WIDTH]);
void fft_shift(complex_fixed_t input[IMAGE_HEIGHT][IMAGE_WIDTH],
               complex_fixed_t output[IMAGE_HEIGHT][IMAGE_WIDTH]);
void spectral_relu(complex_fixed_t* data, int height, int width, int channels);
void spectral_pooling(complex_fixed_t* input, complex_fixed_t* output,
                      int in_height, int in_width, int channels);
void batch_norm(complex_fixed_t* data,
                fixed_t gamma[], fixed_t beta[],
                fixed_t mean[], fixed_t variance[],
                int height, int width, int channels,
                bool is_real);

// Accepts fixed_t weights, biases, and batch-norm parameters
void spectral_conv_layer(complex_fixed_t* input,
                         complex_fixed_t* output,
                         fixed_t* weights_real, fixed_t* weights_imag,
                         fixed_t* bias_real, fixed_t* bias_imag,
                         fixed_t* bn_real_gamma, fixed_t* bn_real_beta,
                         fixed_t* bn_real_mean, fixed_t* bn_real_var,
                         fixed_t* bn_imag_gamma, fixed_t* bn_imag_beta,
                         fixed_t* bn_imag_mean, fixed_t* bn_imag_var,
                         int height, int width, int in_channels, int out_channels);

void fully_connected(complex_fixed_t* input,
                    fixed_t output[NUM_CLASSES],
                    int height, int width, int channels);

// Accepts a fixed_t image and produces fixed_t logits
void spcnn_inference(fixed_t input_image[IMAGE_HEIGHT][IMAGE_WIDTH],
                     fixed_t output_logits[NUM_CLASSES]);

void softmax(fixed_t input[NUM_CLASSES], fixed_t output[NUM_CLASSES]);
int get_prediction(fixed_t logits[NUM_CLASSES]);

// ============================================================
// Utility functions for fixed-point arithmetic
// ============================================================

// Complex multiplication
inline complex_fixed_t complex_mult(complex_fixed_t a, complex_fixed_t b) {
    #pragma HLS INLINE
    fixed_t real_part = a.real() * b.real() - a.imag() * b.imag();
    fixed_t imag_part = a.real() * b.imag() + a.imag() * b.real();
    return complex_fixed_t(real_part, imag_part);
}

// Fixed-point division with scaling
inline fixed_t fixed_div(fixed_t a, fixed_t b) {
    #pragma HLS INLINE
    // Compare against a fixed_t zero (not an int/float literal) to avoid
    // any possibility of ambiguous overload resolution.
    if (b == (fixed_t)0) return (fixed_t)0;
    return a / b;
}

// Fast approximate square root for fixed-point
inline fixed_t fixed_sqrt(fixed_t x) {
    #pragma HLS INLINE
    if (x <= (fixed_t)0) return (fixed_t)0;
    // Single explicit cast to float before calling hls::sqrt, then cast
    // the float result back to fixed_t explicitly (no implicit dual-path
    // conversion left for the compiler to resolve).
    float xf = (float)x;
    float rf = hls::sqrt(xf);
    return (fixed_t)rf;
}

// Fixed-point exponential approximation for softmax
inline fixed_t fixed_exp(fixed_t x) {
    #pragma HLS INLINE
    float xf = (float)x;
    float rf = hls::exp(xf);
    return (fixed_t)rf;
}

// Complex number magnitude for fixed-point
inline fixed_t complex_mag(complex_fixed_t c) {
    #pragma HLS INLINE
    fixed_t real_sq = c.real() * c.real();
    fixed_t imag_sq = c.imag() * c.imag();
    return fixed_sqrt(real_sq + imag_sq);
}

#endif // INFERENCE_H