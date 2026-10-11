#include "inference.h"
#include "model_weights_94class.h"
#include <cstdint>

// ============================================================================
// FFT Implementation
// ============================================================================

void fft_1d(complex_fixed_t* __restrict__ input,
            complex_fixed_t* __restrict__ output,
            int N, bool inverse = false) {
	#pragma HLS ALLOCATION operation instances=mul limit=2
	#pragma HLS ALLOCATION operation instances=add limit=4
    if (N == 1) {
        output[0] = input[0];
        return;
    }

    // Bit-reversal permutation
    for (int i = 0; i < N; i++) {
        #pragma HLS PIPELINE
        int j = 0, temp = i, log2N = 0, tmp = N;
        while (tmp > 1) { tmp >>= 1; log2N++; }
        for (int k = 0; k < log2N; k++) { j = (j << 1) | (temp & 1); temp >>= 1; }
        if (j > i) {
            complex_fixed_t swap = input[i];
            output[i] = input[j];
            output[j] = swap;
        } else {
            output[i] = input[i];
        }
    }

    // Cooley-Tukey butterfly stages
    for (int s = 1; s <= hls::log2((float)N); s++) {
        int m  = 1 << s;
        int m2 = m >> 1;
        float angle = (inverse ? 2.0f : -2.0f) * PI / m;
        fixed_t cos_val = (fixed_t)hls::cos(angle);
        fixed_t sin_val = (fixed_t)hls::sin(angle);
        complex_fixed_t wm(cos_val, sin_val);
        complex_fixed_t w((fixed_t)1.0, (fixed_t)0.0);

        for (int j = 0; j < m2; j++) {
            for (int k = j; k < N; k += m) {
                #pragma HLS PIPELINE
                complex_fixed_t t = complex_mult(w, output[k + m2]);
                complex_fixed_t u = output[k];
                output[k]      = u + t;
                output[k + m2] = u - t;
            }
            w = complex_mult(w, wm);
        }
    }

    if (inverse) {
        fixed_t scale = (fixed_t)(1.0f / N);
        for (int i = 0; i < N; i++) {
            #pragma HLS PIPELINE
            output[i] = complex_fixed_t(output[i].real() * scale,
                                        output[i].imag() * scale);
        }
    }
}

// ============================================================================
// 2D FFT
// ============================================================================

void fft_2d(complex_fixed_t input[IMAGE_HEIGHT][IMAGE_WIDTH],
            complex_fixed_t output[IMAGE_HEIGHT][IMAGE_WIDTH]) {
    #pragma HLS INLINE off

    complex_fixed_t temp[IMAGE_HEIGHT][IMAGE_WIDTH];
    complex_fixed_t row[IMAGE_WIDTH];
    complex_fixed_t col[IMAGE_HEIGHT];
	#pragma HLS ARRAY_PARTITION variable=temp type=cyclic factor=16 dim=1
    // Row-wise FFT
    for (int i = 0; i < IMAGE_HEIGHT; i++) {
        for (int j = 0; j < IMAGE_WIDTH; j++) {
            #pragma HLS PIPELINE
            row[j] = input[i][j];
        }
        fft_1d(row, row, IMAGE_WIDTH);
        for (int j = 0; j < IMAGE_WIDTH; j++) {
            #pragma HLS PIPELINE
            temp[i][j] = row[j];
        }
    }

    // Column-wise FFT
    for (int j = 0; j < IMAGE_WIDTH; j++) {
        for (int i = 0; i < IMAGE_HEIGHT; i++) {
            #pragma HLS PIPELINE 
            col[i] = temp[i][j];
        }
        fft_1d(col, col, IMAGE_HEIGHT);
        for (int i = 0; i < IMAGE_HEIGHT; i++) {
            #pragma HLS PIPELINE
            output[i][j] = col[i];
        }
    }
}

// ============================================================================
// FFT Shift
// ============================================================================

void fft_shift(complex_fixed_t input[IMAGE_HEIGHT][IMAGE_WIDTH],
               complex_fixed_t output[IMAGE_HEIGHT][IMAGE_WIDTH]) {
    #pragma HLS INLINE off

    int h_half = IMAGE_HEIGHT / 2;
    int w_half = IMAGE_WIDTH  / 2;

    for (int i = 0; i < IMAGE_HEIGHT; i++) {
        for (int j = 0; j < IMAGE_WIDTH; j++) {
            #pragma HLS PIPELINE
            int new_i = (i + h_half) % IMAGE_HEIGHT;
            int new_j = (j + w_half) % IMAGE_WIDTH;
            output[new_i][new_j] = input[i][j];
        }
    }
}

// ============================================================================
// Spectral ReLU
// ============================================================================

void spectral_relu(complex_fixed_t* __restrict__ data,
                   int height, int width, int channels) {
    #pragma HLS INLINE off
    #pragma HLS ALLOCATION operation instances=mul limit=2
 
    const fixed_t threshold_sq = (fixed_t)(0.1f * 0.1f);

    for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
            for (int c = 0; c < channels; c++) {
                #pragma HLS PIPELINE II=1
                int    idx           = (h * width + w) * MAX_CHANNELS + c;
                fixed_t real_part    = data[idx].real();
                fixed_t imag_part    = data[idx].imag();
                fixed_t magnitude_sq = real_part * real_part + imag_part * imag_part;
                if (magnitude_sq <= threshold_sq) {
                    data[idx] = complex_fixed_t((fixed_t)0.0f, (fixed_t)0.0f);
                }
            }
        }
    }
}

// ============================================================================
// Spectral Pooling
// ============================================================================

void spectral_pooling(complex_fixed_t* __restrict__ input,
                      complex_fixed_t* __restrict__ output,
                      int in_height, int in_width, int channels) {
    #pragma HLS INLINE off

    int start_h    = in_height / 4;
    int start_w    = in_width  / 4;
    int out_height = in_height / 2;
    int out_width  = in_width  / 2;

    for (int h = 0; h < out_height; h++) {
        for (int w = 0; w < out_width; w++) {
            for (int c = 0; c < channels; c++) {
                #pragma HLS PIPELINE
                int in_idx  = ((start_h + h) * in_width  + (start_w + w)) * MAX_CHANNELS + c;
                int out_idx = (h * out_width + w) * MAX_CHANNELS + c;
                output[out_idx] = input[in_idx];
            }
        }
    }
}

// ============================================================================
// Batch Normalization
// ============================================================================

void batch_norm(complex_fixed_t* __restrict__ data,
                const float gamma[], const float beta[],
                const float mean[],  const float variance[],
                int height, int width, int channels,
                bool is_real) {
    #pragma HLS INLINE off
    #pragma HLS ALLOCATION operation instances=mul limit=2
	#pragma HLS ALLOCATION operation instances=fixed_sqrt limit=1
    #pragma HLS ALLOCATION operation instances=fixed_div limit=1
    const fixed_t epsilon = (fixed_t)1e-3f;

    for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
            for (int c = 0; c < channels; c++) {
                #pragma HLS PIPELINE
                int     idx      = (h * width + w) * MAX_CHANNELS + c;
                fixed_t value    = is_real ? data[idx].real() : data[idx].imag();

                fixed_t var_fixed   = (fixed_t)variance[c];
                fixed_t mean_fixed  = (fixed_t)mean[c];
                fixed_t gamma_fixed = (fixed_t)gamma[c];
                fixed_t beta_fixed  = (fixed_t)beta[c];

                if (var_fixed < (fixed_t)0.0f) var_fixed = (fixed_t)0.0f;

                fixed_t std_val    = fixed_sqrt(var_fixed + epsilon);
                fixed_t normalized = fixed_div(value - mean_fixed, std_val);
                fixed_t out_val    = gamma_fixed * normalized + beta_fixed;

                if (is_real) {
                    data[idx] = complex_fixed_t(out_val, data[idx].imag());
                } else {
                    data[idx] = complex_fixed_t(data[idx].real(), out_val);
                }
            }
        }
    }
}

// ============================================================================
// Spectral Convolution Layer
// ============================================================================

void spectral_conv_layer(complex_fixed_t* __restrict__ input,
                         complex_fixed_t* __restrict__ output,
                         const float* __restrict__ weights_real,
                         const float* __restrict__ weights_imag,
                         const float* __restrict__ bias_real,
                         const float* __restrict__ bias_imag,
                         const float* __restrict__ bn_real_gamma,
                         const float* __restrict__ bn_real_beta,
                         const float* __restrict__ bn_real_mean,
                         const float* __restrict__ bn_real_var,
                         const float* __restrict__ bn_imag_gamma,
                         const float* __restrict__ bn_imag_beta,
                         const float* __restrict__ bn_imag_mean,
                         const float* __restrict__ bn_imag_var,
                         int height, int width, int in_channels, int out_channels) {
    #pragma HLS INLINE off
    #pragma HLS ALLOCATION operation instances=mul limit=4
    // 4 multiplier instances: complex multiply (ac-bd, ad+bc) needs exactly
    // 4 real muls per cycle at II=1 on the in_c loop. Fewer forces II>1.

    // Initialize output buffer
    for (int i = 0; i < height * width * MAX_CHANNELS; i++) {
        #pragma HLS PIPELINE
        output[i] = complex_fixed_t((fixed_t)0.0f, (fixed_t)0.0f);
    }

    // Spectral (Hadamard) product with weight tensor
    for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
            for (int out_c = 0; out_c < out_channels; out_c++) {
                fixed_t sum_real = (fixed_t)bias_real[out_c];
                fixed_t sum_imag = (fixed_t)bias_imag[out_c];

                for (int in_c = 0; in_c < in_channels; in_c++) {
                    #pragma HLS PIPELINE

                    int input_idx  = (h * width + w) * MAX_CHANNELS + in_c;
                    int weight_idx = ((h * width + w) * in_channels + in_c) * out_channels + out_c;

                    fixed_t in_real = input[input_idx].real();
                    fixed_t in_imag = input[input_idx].imag();
                    fixed_t w_real  = (fixed_t)weights_real[weight_idx];
                    fixed_t w_imag  = (fixed_t)weights_imag[weight_idx];

                    sum_real += in_real * w_real - in_imag * w_imag;
                    sum_imag += in_real * w_imag + in_imag * w_real;
                }

                int output_idx = (h * width + w) * MAX_CHANNELS + out_c;
                output[output_idx] = complex_fixed_t(sum_real, sum_imag);
            }
        }
    }

    batch_norm(output, bn_real_gamma, bn_real_beta, bn_real_mean, bn_real_var,
               height, width, out_channels, true);
    batch_norm(output, bn_imag_gamma, bn_imag_beta, bn_imag_mean, bn_imag_var,
               height, width, out_channels, false);

    spectral_relu(output, height, width, out_channels);
}

// ============================================================================
// Fully Connected Layer
// ============================================================================

void fully_connected(complex_fixed_t* __restrict__ input,
                     fixed_t output[NUM_CLASSES],
                     int height, int width, int channels) {
    #pragma HLS INLINE off

    int feature_length = height * width * channels;
    const int MAX_FEATURES = 1024;
    fixed_t features[MAX_FEATURES];
	#pragma HLS BIND_STORAGE variable=features type=ram_2p impl=lutram

    // Zero-initialise
    for (int i = 0; i < MAX_FEATURES; i++) {
        #pragma HLS PIPELINE
        features[i] = (fixed_t)0.0f;
    }

    // Flatten: magnitude-squared of each complex feature
    for (int i = 0; i < feature_length; i++) {
        #pragma HLS PIPELINE 
		#pragma HLS DEPENDENCE variable=features inter false
        int h   = i / (width * channels);
        int w   = (i / channels) % width;
        int c   = i % channels;
        int idx = (h * width + w) * MAX_CHANNELS + c;
        fixed_t re = input[idx].real();
        fixed_t im = input[idx].imag();
        features[i] = re * re + im * im;
    }

    // Output initialisation
    for (int c = 0; c < NUM_CLASSES; c++) {
        #pragma HLS PIPELINE
        output[c] = (fixed_t)dense_bias[c];
    }

    // Matrix–vector multiply
    for (int c = 0; c < NUM_CLASSES; c++) {
        fixed_t sum = output[c];
        for (int i = 0; i < feature_length; i++) {
            #pragma HLS PIPELINE
			#pragma HLS DEPENDENCE variable=features inter false
            fixed_t weight = (fixed_t)dense_weights[i * NUM_CLASSES + c];
            sum += features[i] * weight;
        }
        output[c] = sum;
    }
}

// ============================================================================
// Main Inference
// ============================================================================

void spcnn_inference(float input_image[IMAGE_HEIGHT][IMAGE_WIDTH],
                     fixed_t output_logits[NUM_CLASSES]) {
    #pragma HLS DATAFLOW 
    complex_fixed_t buffer1[MAX_HEIGHT * MAX_WIDTH * MAX_CHANNELS];
    complex_fixed_t buffer2[MAX_HEIGHT * MAX_WIDTH * MAX_CHANNELS];
    #pragma HLS ARRAY_PARTITION variable=buffer1 type=cyclic factor=8 dim=1
	#pragma HLS ARRAY_PARTITION variable=buffer2 type=cyclic factor=8 dim=1

    for (int i = 0; i < MAX_HEIGHT * MAX_WIDTH * MAX_CHANNELS; i++) {
        
        const complex_fixed_t zero((fixed_t)0.0f, (fixed_t)0.0f);
        buffer1[i] = zero;
        buffer2[i] = zero;
    }

    // Convert float input ? complex fixed-point
    complex_fixed_t fft_input[IMAGE_HEIGHT][IMAGE_WIDTH];
    complex_fixed_t fft_output[IMAGE_HEIGHT][IMAGE_WIDTH];

    // CHANGED ram_2p ? ram_1p: written once then read once, strictly serial.
    #pragma HLS BIND_STORAGE variable=fft_input  type=ram_1p impl=bram
    #pragma HLS BIND_STORAGE variable=fft_output type=ram_1p impl=bram

    for (int i = 0; i < IMAGE_HEIGHT; i++) {
        for (int j = 0; j < IMAGE_WIDTH; j++) {
            #pragma HLS PIPELINE II=1
            fft_input[i][j] = complex_fixed_t((fixed_t)input_image[i][j], (fixed_t)0.0f);
        }
    }

    // 2-D FFT + shift
    fft_2d(fft_input, fft_output);

    complex_fixed_t shifted[IMAGE_HEIGHT][IMAGE_WIDTH];

    // CHANGED ram_2p ? ram_1p: fft_shift writes, copy loop reads — serial.
    #pragma HLS BIND_STORAGE variable=shifted type=ram_1p impl=bram
    fft_shift(fft_output, shifted);

    // Copy shifted spectrum into channel-major buffer
    for (int i = 0; i < IMAGE_HEIGHT; i++) {
        for (int j = 0; j < IMAGE_WIDTH; j++) {
            #pragma HLS PIPELINE 
            int idx = (i * IMAGE_WIDTH + j) * MAX_CHANNELS;
            buffer1[idx] = shifted[i][j];
        }
    }

    // Layer 1 — 64×64, 1?2 channels
    spectral_conv_layer(buffer1, buffer2,
                        conv1_real, conv1_imag,
                        conv1_bias_real, conv1_bias_imag,
                        bn1_real_gamma, bn1_real_beta, bn1_real_mean, bn1_real_var,
                        bn1_imag_gamma, bn1_imag_beta, bn1_imag_mean, bn1_imag_var,
                        64, 64, 1, 2);
    spectral_pooling(buffer2, buffer1, 64, 64, 2);

    // Layer 2 — 32×32, 2?4 channels
    spectral_conv_layer(buffer1, buffer2,
                        conv2_real, conv2_imag,
                        conv2_bias_real, conv2_bias_imag,
                        bn2_real_gamma, bn2_real_beta, bn2_real_mean, bn2_real_var,
                        bn2_imag_gamma, bn2_imag_beta, bn2_imag_mean, bn2_imag_var,
                        32, 32, 2, 4);
    spectral_pooling(buffer2, buffer1, 32, 32, 4);

    // Layer 3 — 16×16, 4?16 channels
    spectral_conv_layer(buffer1, buffer2,
                        conv3_real, conv3_imag,
                        conv3_bias_real, conv3_bias_imag,
                        bn3_real_gamma, bn3_real_beta, bn3_real_mean, bn3_real_var,
                        bn3_imag_gamma, bn3_imag_beta, bn3_imag_mean, bn3_imag_var,
                        16, 16, 4, 16);
    spectral_pooling(buffer2, buffer1, 16, 16, 16);

    // Classifier
    fully_connected(buffer1, output_logits, 8, 8, 16);
}

// ============================================================================
// Softmax
// ============================================================================

void softmax(fixed_t input[NUM_CLASSES], fixed_t output[NUM_CLASSES]) {
    #pragma HLS INLINE off

    fixed_t max_val = input[0];
    for (int i = 1; i < NUM_CLASSES; i++) {
        #pragma HLS PIPELINE 
        if (input[i] > max_val) max_val = input[i];
    }

    fixed_t sum = (fixed_t)0.0f;
    for (int i = 0; i < NUM_CLASSES; i++) {
        #pragma HLS PIPELINE
        output[i] = fixed_exp(input[i] - max_val);
        sum += output[i];
    }

    for (int i = 0; i < NUM_CLASSES; i++) {
        #pragma HLS PIPELINE 
        output[i] = fixed_div(output[i], sum);
    }
}

// ============================================================================
// Get Prediction
// ============================================================================

int get_prediction(fixed_t logits[NUM_CLASSES]) {
    #pragma HLS INLINE off

    int     max_idx = 0;
    fixed_t max_val = logits[0];

    for (int i = 1; i < NUM_CLASSES; i++) {
        #pragma HLS PIPELINE
        if (logits[i] > max_val) {
            max_val = logits[i];
            max_idx = i;
        }
    }
    return max_idx;
}
