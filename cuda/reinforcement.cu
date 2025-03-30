/**
 * CUDA-AI-Nintendo_NES_Agent - Reinforcement Learning CUDA Kernels
 * 
 * This file contains CUDA kernel implementations for reinforcement learning
 * operations used by the AI agent. It provides optimized GPU implementations
 * of various RL algorithms including Q-learning, experience replay, and
 * policy gradient methods.
 * 
 * These kernels are specifically optimized for training agents to play
 * NES games with high performance.
 * 
 * Author: doublegate
 * License: MIT
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <curand_kernel.h>

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
 * CUDA kernel for batched Q-learning updates
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
 */
__global__ void q_learning_update_kernel(float* q_values, const float* target_q_values,
                                        const int* actions, const float* rewards,
                                        const bool* dones, float gamma, float learning_rate,
                                        int batch_size, int action_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < batch_size) {
        // Get the current Q-value for the action taken
        int action = actions[idx];
        float current_q = q_values[idx * action_dim + action];
        
        // Find the maximum target Q-value for the next state
        float max_target_q = -INFINITY;
        for (int a = 0; a < action_dim; ++a) {
            float target_q = target_q_values[idx * action_dim + a];
            max_target_q = fmaxf(max_target_q, target_q);
        }
        
        // Calculate the TD target
        float td_target;
        if (dones[idx]) {
            td_target = rewards[idx];
        } else {
            td_target = rewards[idx] + gamma * max_target_q;
        }
        
        // Calculate the TD error
        float td_error = td_target - current_q;
        
        // Update the Q-value
        q_values[idx * action_dim + action] += learning_rate * td_error;
    }
}

/**
 * CUDA kernel for epsilon-greedy action selection
 * 
 * @param q_values Q-values for each action
 * @param actions Selected actions
 * @param epsilon Exploration rate
 * @param batch_size Batch size
 * @param action_dim Number of possible actions
 * @param seeds Random seeds
 */
__global__ void epsilon_greedy_kernel(const float* q_values, int* actions,
                                     float epsilon, int batch_size, int action_dim,
                                     unsigned long long* seeds) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < batch_size) {
        // Initialize random state
        curandState state;
        curand_init(seeds[idx], 0, 0, &state);
        
        // Decide whether to explore or exploit
        float r = curand_uniform(&state);
        
        if (r < epsilon) {
            // Explore: choose a random action
            actions[idx] = (int)(curand_uniform(&state) * action_dim);
        } else {
            // Exploit: choose the best action
            float max_q = -INFINITY;
            int best_action = 0;
            
            for (int a = 0; a < action_dim; ++a) {
                float q = q_values[idx * action_dim + a];
                if (q > max_q) {
                    max_q = q;
                    best_action = a;
                }
            }
            
            actions[idx] = best_action;
        }
    }
}

/**
 * CUDA kernel for computing n-step returns
 * 
 * @param rewards Rewards for each step
 * @param dones Terminal state flags
 * @param values Value function estimates
 * @param returns Computed returns
 * @param gamma Discount factor
 * @param n_steps Number of steps
 * @param batch_size Batch size
 */
__global__ void compute_n_step_returns_kernel(const float* rewards, const bool* dones,
                                            const float* values, float* returns,
                                            float gamma, int n_steps, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < batch_size) {
        float G = 0.0f;
        
        // Start from the end and work backwards
        for (int i = n_steps - 1; i >= 0; --i) {
            int step_idx = idx * n_steps + i;
            
            if (i == n_steps - 1 && !dones[step_idx]) {
                // Bootstrap with value function if not terminal
                G = values[step_idx];
            }
            
            if (dones[step_idx]) {
                // Terminal state
                G = rewards[step_idx];
            } else {
                // Non-terminal state
                G = rewards[step_idx] + gamma * G;
            }
            
            returns[step_idx] = G;
        }
    }
}

/**
 * CUDA kernel for computing TD(lambda) returns
 * 
 * @param rewards Rewards for each step
 * @param values Value function estimates
 * @param dones Terminal state flags
 * @param advantages Computed advantages
 * @param gamma Discount factor
 * @param lambda Lambda parameter
 * @param batch_size Batch size
 * @param n_steps Number of steps
 */
__global__ void compute_gae_kernel(const float* rewards, const float* values,
                                 const bool* dones, float* advantages,
                                 float gamma, float lambda, int batch_size, int n_steps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < batch_size) {
        float gae = 0.0f;
        
        for (int i = n_steps - 1; i >= 0; --i) {
            int step_idx = idx * n_steps + i;
            int next_idx = step_idx + 1;
            
            float delta;
            if (i == n_steps - 1 || dones[step_idx]) {
                delta = rewards[step_idx] - values[step_idx];
            } else {
                delta = rewards[step_idx] + gamma * values[next_idx] - values[step_idx];
            }
            
            gae = delta + gamma * lambda * (dones[step_idx] ? 0.0f : gae);
            advantages[step_idx] = gae;
        }
    }
}

/**
 * CUDA kernel for prioritized experience replay
 * 
 * @param priorities Experience priorities
 * @param errors TD errors
 * @param alpha Priority exponent
 * @param epsilon Small constant for stability
 * @param batch_size Batch size
 */
__global__ void update_priorities_kernel(float* priorities, const float* errors,
                                       float alpha, float epsilon, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < batch_size) {
        float abs_error = fabsf(errors[idx]);
        priorities[idx] = powf(abs_error + epsilon, alpha);
    }
}

// C wrapper functions for each kernel to be called from Rust

extern "C" {

/**
 * Update Q-values using Q-learning
 */
cudaError_t q_learning_update_cuda(float* q_values, const float* target_q_values,
                                  const int* actions, const float* rewards,
                                  const bool* dones, float gamma, float learning_rate,
                                  int batch_size, int action_dim) {
    dim3 grid, block;
    get_grid_block_dims(batch_size, grid, block);
    
    q_learning_update_kernel<<<grid, block>>>(q_values, target_q_values, actions, rewards,
                                            dones, gamma, learning_rate, batch_size, action_dim);
    return cudaGetLastError();
}

/**
 * Select actions using epsilon-greedy policy
 */
cudaError_t epsilon_greedy_cuda(const float* q_values, int* actions,
                              float epsilon, int batch_size, int action_dim,
                              unsigned long long* seeds) {
    dim3 grid, block;
    get_grid_block_dims(batch_size, grid, block);
    
    epsilon_greedy_kernel<<<grid, block>>>(q_values, actions, epsilon, batch_size, action_dim, seeds);
    return cudaGetLastError();
}

/**
 * Compute n-step returns
 */
cudaError_t compute_n_step_returns_cuda(const float* rewards, const bool* dones,
                                      const float* values, float* returns,
                                      float gamma, int n_steps, int batch_size) {
    dim3 grid, block;
    get_grid_block_dims(batch_size, grid, block);
    
    compute_n_step_returns_kernel<<<grid, block>>>(rewards, dones, values, returns,
                                                 gamma, n_steps, batch_size);
    return cudaGetLastError();
}

/**
 * Compute generalized advantage estimation (GAE)
 */
cudaError_t compute_gae_cuda(const float* rewards, const float* values,
                           const bool* dones, float* advantages,
                           float gamma, float lambda, int batch_size, int n_steps) {
    dim3 grid, block;
    get_grid_block_dims(batch_size, grid, block);
    
    compute_gae_kernel<<<grid, block>>>(rewards, values, dones, advantages,
                                      gamma, lambda, batch_size, n_steps);
    return cudaGetLastError();
}

/**
 * Update priorities for prioritized experience replay
 */
cudaError_t update_priorities_cuda(float* priorities, const float* errors,
                                 float alpha, float epsilon, int batch_size) {
    dim3 grid, block;
    get_grid_block_dims(batch_size, grid, block);
    
    update_priorities_kernel<<<grid, block>>>(priorities, errors, alpha, epsilon, batch_size);
    return cudaGetLastError();
}

} // extern "C"
