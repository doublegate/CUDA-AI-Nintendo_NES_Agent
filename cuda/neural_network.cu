/**
 * CUDA-AI-Nintendo_NES_Agent - Neural Network CUDA Kernels
 * 
 * This file contains CUDA kernel implementations for neural network operations
 * used by the AI agent. It provides optimized GPU implementations of various
 * deep learning operations including convolution, activation functions, and
 * backpropagation.
 * 
 * These kernels are specifically optimized for the types of neural networks
 * useful in NES game playing, focusing on processing pixel data efficiently.
 * 
 * Author: doublegate
 * License: MIT
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

// Error checking macro
#define CUDA_CHECK(call) \
do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(error)); \
        return error; \
    } \
} while(0)

// Utility function to get grid and block sizes
inline void get_grid_block_dims(int n, dim3 &grid, dim3 &block) {
    block.x = 256;
    grid.x = (n + block.x - 1) / block.x;
}

/**
 * CUDA kernel for ReLU activation function
 * 
 * @param input The input tensor
 * @param output The output tensor
 * @param n The number of elements
 */
__global__ void relu_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

/**
 * CUDA kernel for ReLU activation function gradient
 * 
 * @param input The input tensor
 * @param grad_output The gradient from the next layer
 * @param grad_input The gradient to be calculated
 * @param n The number of elements
 */
__global__ void relu_gradient_kernel(const float* input, const float* grad_output, 
                                    float* grad_input, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        grad_input[idx] = input[idx] > 0.0f ? grad_output[idx] : 0.0f;
    }
}

/**
 * CUDA kernel for 2D convolution forward pass
 * 
 * @param input The input tensor (NCHW format)
 * @param weight The weight tensor
 * @param bias The bias tensor
 * @param output The output tensor
 * @param batch_size Batch size
 * @param in_channels Input channels
 * @param out_channels Output channels
 * @param in_height Input height
 * @param in_width Input width
 * @param kernel_height Kernel height
 * @param kernel_width Kernel width
 * @param stride_h Stride height
 * @param stride_w Stride width
 * @param padding_h Padding height
 * @param padding_w Padding width
 */
__global__ void conv2d_forward_kernel(const float* input, const float* weight, const float* bias,
                                     float* output, int batch_size, int in_channels, int out_channels,
                                     int in_height, int in_width, int kernel_height, int kernel_width,
                                     int stride_h, int stride_w, int padding_h, int padding_w) {
    // Calculate output dimensions
    int out_height = (in_height + 2 * padding_h - kernel_height) / stride_h + 1;
    int out_width = (in_width + 2 * padding_w - kernel_width) / stride_w + 1;
    
    // Get global thread indices
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Calculate output indices
    int n = idx / (out_channels * out_height * out_width);
    int c = (idx / (out_height * out_width)) % out_channels;
    int h = (idx / out_width) % out_height;
    int w = idx % out_width;
    
    if (n < batch_size && c < out_channels && h < out_height && w < out_width) {
        float sum = bias[c];
        
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int kh = 0; kh < kernel_height; ++kh) {
                for (int kw = 0; kw < kernel_width; ++kw) {
                    int h_in = h * stride_h - padding_h + kh;
                    int w_in = w * stride_w - padding_w + kw;
                    
                    if (h_in >= 0 && h_in < in_height && w_in >= 0 && w_in < in_width) {
                        int input_idx = ((n * in_channels + ic) * in_height + h_in) * in_width + w_in;
                        int weight_idx = ((c * in_channels + ic) * kernel_height + kh) * kernel_width + kw;
                        
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        
        output[idx] = sum;
    }
}

/**
 * CUDA kernel for max pooling forward pass
 * 
 * @param input The input tensor (NCHW format)
 * @param output The output tensor
 * @param index The indices of max values
 * @param batch_size Batch size
 * @param channels Channels
 * @param in_height Input height
 * @param in_width Input width
 * @param kernel_height Kernel height
 * @param kernel_width Kernel width
 * @param stride_h Stride height
 * @param stride_w Stride width
 * @param padding_h Padding height
 * @param padding_w Padding width
 */
__global__ void max_pool2d_forward_kernel(const float* input, float* output, int* index,
                                         int batch_size, int channels, int in_height, int in_width,
                                         int kernel_height, int kernel_width, int stride_h, int stride_w,
                                         int padding_h, int padding_w) {
    // Calculate output dimensions
    int out_height = (in_height + 2 * padding_h - kernel_height) / stride_h + 1;
    int out_width = (in_width + 2 * padding_w - kernel_width) / stride_w + 1;
    
    // Get global thread indices
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Calculate output indices
    int n = idx / (channels * out_height * out_width);
    int c = (idx / (out_height * out_width)) % channels;
    int h = (idx / out_width) % out_height;
    int w = idx % out_width;
    
    if (n < batch_size && c < channels && h < out_height && w < out_width) {
        float max_val = -INFINITY;
        int max_idx = -1;
        
        for (int kh = 0; kh < kernel_height; ++kh) {
            for (int kw = 0; kw < kernel_width; ++kw) {
                int h_in = h * stride_h - padding_h + kh;
                int w_in = w * stride_w - padding_w + kw;
                
                if (h_in >= 0 && h_in < in_height && w_in >= 0 && w_in < in_width) {
                    int input_idx = ((n * channels + c) * in_height + h_in) * in_width + w_in;
                    float val = input[input_idx];
                    
                    if (val > max_val) {
                        max_val = val;
                        max_idx = input_idx;
                    }
                }
            }
        }
        
        output[idx] = max_val;
        index[idx] = max_idx;
    }
}

/**
 * CUDA kernel for frame preprocessing
 * - Converts RGB to grayscale
 * - Normalizes pixel values to [0, 1]
 * 
 * @param input The input frame (RGB format)
 * @param output The output frame (grayscale, normalized)
 * @param width The frame width
 * @param height The frame height
 */
__global__ void preprocess_frame_kernel(const unsigned char* input, float* output,
                                       int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int x = idx % width;
    int y = idx / width;
    
    if (x < width && y < height) {
        int rgb_offset = (y * width + x) * 3;
        float r = static_cast<float>(input[rgb_offset]);
        float g = static_cast<float>(input[rgb_offset + 1]);
        float b = static_cast<float>(input[rgb_offset + 2]);
        
        // Convert to grayscale and normalize to [0, 1]
        float gray = (0.299f * r + 0.587f * g + 0.114f * b) / 255.0f;
        output[y * width + x] = gray;
    }
}

// C wrapper functions for each kernel to be called from Rust

extern "C" {

/**
 * Apply ReLU activation function
 */
cudaError_t relu_cuda(const float* input, float* output, int n) {
    dim3 grid, block;
    get_grid_block_dims(n, grid, block);
    
    relu_kernel<<<grid, block>>>(input, output, n);
    return cudaGetLastError();
}

/**
 * Apply ReLU gradient
 */
cudaError_t relu_gradient_cuda(const float* input, const float* grad_output, 
                              float* grad_input, int n) {
    dim3 grid, block;
    get_grid_block_dims(n, grid, block);
    
    relu_gradient_kernel<<<grid, block>>>(input, grad_output, grad_input, n);
    return cudaGetLastError();
}

/**
 * Apply 2D convolution
 */
cudaError_t conv2d_forward_cuda(const float* input, const float* weight, const float* bias,
                               float* output, int batch_size, int in_channels, int out_channels,
                               int in_height, int in_width, int kernel_height, int kernel_width,
                               int stride_h, int stride_w, int padding_h, int padding_w) {
    // Calculate output dimensions
    int out_height = (in_height + 2 * padding_h - kernel_height) / stride_h + 1;
    int out_width = (in_width + 2 * padding_w - kernel_width) / stride_w + 1;
    int n = batch_size * out_channels * out_height * out_width;
    
    dim3 grid, block;
    get_grid_block_dims(n, grid, block);
    
    conv2d_forward_kernel<<<grid, block>>>(input, weight, bias, output, batch_size, in_channels, 
                                          out_channels, in_height, in_width, kernel_height, 
                                          kernel_width, stride_h, stride_w, padding_h, padding_w);
    return cudaGetLastError();
}

/**
 * Apply max pooling
 */
cudaError_t max_pool2d_forward_cuda(const float* input, float* output, int* index,
                                   int batch_size, int channels, int in_height, int in_width,
                                   int kernel_height, int kernel_width, int stride_h, int stride_w,
                                   int padding_h, int padding_w) {
    // Calculate output dimensions
    int out_height = (in_height + 2 * padding_h - kernel_height) / stride_h + 1;
    int out_width = (in_width + 2 * padding_w - kernel_width) / stride_w + 1;
    int n = batch_size * channels * out_height * out_width;
    
    dim3 grid, block;
    get_grid_block_dims(n, grid, block);
    
    max_pool2d_forward_kernel<<<grid, block>>>(input, output, index, batch_size, channels, 
                                              in_height, in_width, kernel_height, kernel_width, 
                                              stride_h, stride_w, padding_h, padding_w);
    return cudaGetLastError();
}

/**
 * Preprocess a frame
 */
cudaError_t preprocess_frame_cuda(const unsigned char* input, float* output,
                                 int width, int height) {
    int n = width * height;
    
    dim3 grid, block;
    get_grid_block_dims(n, grid, block);
    
    preprocess_frame_kernel<<<grid, block>>>(input, output, width, height);
    return cudaGetLastError();
}

} // extern "C"
