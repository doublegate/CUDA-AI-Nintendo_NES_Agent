/**
 * CUDA-AI-Nintendo_NES_Agent - CUDA Interface Header
 * 
 * This header file defines the C interface to the CUDA kernels used by the
 * AI agent. It provides function declarations for neural network operations,
 * reinforcement learning algorithms, and utility functions.
 * 
 * This interface is designed to be called from Rust code using FFI.
 * 
 * Author: doublegate
 * License: MIT
 */

#ifndef CUDA_INTERFACE_H
#define CUDA_INTERFACE_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

//--------------------------------------------------
// Neural Network Operations
//--------------------------------------------------

/**
 * Apply ReLU activation function
 * 
 * @param input The input tensor
 * @param output The output tensor
 * @param n The number of elements
 * @return CUDA error code
 */
cudaError_t relu_cuda(const float* input, float* output, int n);

/**
 * Apply ReLU gradient
 * 
 * @param input The input tensor
 * @param grad_output The gradient from the next layer
 * @param grad_input The gradient to be calculated
 * @param n The number of elements
 * @return CUDA error code
 */
cudaError_t relu_gradient_cuda(const float* input, const float* grad_output, 
                              float* grad_input, int n);

/**
 * Apply 2D convolution
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
 * @return CUDA error code
 */
cudaError_t conv2d_forward_cuda(const float* input, const float* weight, const float* bias,
                               float* output, int batch_size, int in_channels, int out_channels,
                               int in_height, int in_width, int kernel_height, int kernel_width,
                               int stride_h, int stride_w, int padding_h, int padding_w);

/**
 * Apply max pooling
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
 * @return CUDA error code
 */
cudaError_t max_pool2d_forward_cuda(const float* input, float* output, int* index,
                                   int batch_size, int channels, int in_height, int in_width,
                                   int kernel_height, int kernel_width, int stride_h, int stride_w,
                                   int padding_h, int padding_w);

/**
 * Preprocess a frame
 * 
 * @param input The input frame (RGB format)
 * @param output The output frame (grayscale, normalized)
 * @param width The frame width
 * @param height The frame height
 * @return CUDA error code
 */
cudaError_t preprocess_frame_cuda(const unsigned char* input, float* output,
                                 int width, int height);

//--------------------------------------------------
// Reinforcement Learning Operations
//--------------------------------------------------

/**
 * Update Q-values using Q-learning
 * 
 * @param q_values Current Q-values
 * @param target_q_values Target Q-values
 * @param actions Actions taken
 * @param rewards Rewards received
 * @param dones Terminal state flags
 * @param gamma Discount factor
 * @param learning_rate Learning rate
 * @param batch_size Batch size
 * @param action_dim Number of possible actions
 * @return CUDA error code
 */
cudaError_t q_learning_update_cuda(float* q_values, const float* target_q_values,
                                  const int* actions, const float* rewards,
                                  const bool* dones, float gamma, float learning_rate,
                                  int batch_size, int action_dim);

/**
 * Select actions using epsilon-greedy policy
 * 
 * @param q_values Q-values for each action
 * @param actions Selected actions
 * @param epsilon Exploration rate
 * @param batch_size Batch size
 * @param action_dim Number of possible actions
 * @param seeds Random seeds
 * @return CUDA error code
 */
cudaError_t epsilon_greedy_cuda(const float* q_values, int* actions,
                              float epsilon, int batch_size, int action_dim,
                              unsigned long long* seeds);

/**
 * Compute n-step returns
 * 
 * @param rewards Rewards for each step
 * @param dones Terminal state flags
 * @param values Value function estimates
 * @param returns Computed returns
 * @param gamma Discount factor
 * @param n_steps Number of steps
 * @param batch_size Batch size
 * @return CUDA error code
 */
cudaError_t compute_n_step_returns_cuda(const float* rewards, const bool* dones,
                                      const float* values, float* returns,
                                      float gamma, int n_steps, int batch_size);

/**
 * Compute generalized advantage estimation (GAE)
 * 
 * @param rewards Rewards for each step
 * @param values Value function estimates
 * @param dones Terminal state flags
 * @param advantages Computed advantages
 * @param gamma Discount factor
 * @param lambda Lambda parameter
 * @param batch_size Batch size
 * @param n_steps Number of steps
 * @return CUDA error code
 */
cudaError_t compute_gae_cuda(const float* rewards, const float* values,
                           const bool* dones, float* advantages,
                           float gamma, float lambda, int batch_size, int n_steps);

/**
 * Update priorities for prioritized experience replay
 * 
 * @param priorities Experience priorities
 * @param errors TD errors
 * @param alpha Priority exponent
 * @param epsilon Small constant for stability
 * @param batch_size Batch size
 * @return CUDA error code
 */
cudaError_t update_priorities_cuda(float* priorities, const float* errors,
                                 float alpha, float epsilon, int batch_size);

//--------------------------------------------------
// Memory Management
//--------------------------------------------------

/**
 * Initialize CUDA and get device information
 * 
 * @param device_id Device ID to use (output)
 * @param device_name Device name (output)
 * @param device_memory Device memory in bytes (output)
 * @return CUDA error code
 */
cudaError_t initialize_cuda(int* device_id, char* device_name, size_t* device_memory);

/**
 * Allocate memory on the GPU
 * 
 * @param device_ptr Pointer to allocated memory (output)
 * @param size_bytes Size in bytes
 * @return CUDA error code
 */
cudaError_t cuda_malloc(void** device_ptr, size_t size_bytes);

/**
 * Free memory on the GPU
 * 
 * @param device_ptr Pointer to memory to free
 * @return CUDA error code
 */
cudaError_t cuda_free(void* device_ptr);

/**
 * Copy memory from host to device
 * 
 * @param dst Destination pointer (device)
 * @param src Source pointer (host)
 * @param size_bytes Size in bytes
 * @return CUDA error code
 */
cudaError_t cuda_memcpy_host_to_device(void* dst, const void* src, size_t size_bytes);

/**
 * Copy memory from device to host
 * 
 * @param dst Destination pointer (host)
 * @param src Source pointer (device)
 * @param size_bytes Size in bytes
 * @return CUDA error code
 */
cudaError_t cuda_memcpy_device_to_host(void* dst, const void* src, size_t size_bytes);

/**
 * Set memory on the device
 * 
 * @param device_ptr Pointer to memory to set
 * @param value Value to set
 * @param size_bytes Size in bytes
 * @return CUDA error code
 */
cudaError_t cuda_memset(void* device_ptr, int value, size_t size_bytes);

/**
 * Synchronize CUDA device
 * 
 * @return CUDA error code
 */
cudaError_t cuda_synchronize();

#ifdef __cplusplus
}
#endif

#endif // CUDA_INTERFACE_H
