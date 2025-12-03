#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>
#include <cmath>

// Include the existing CUDA rasterizer headers
#define CUBQL_GPU_BUILDER_IMPLEMENTATION 1
#include <cuBQL/bvh.h>
#include "cuBQL/builder/cuda.h"
#include "cuda_rasterizer/rasterizer.h"
#include "cuda_rasterizer/auxiliary.h"

// Kernel to build bounding boxes for each sample point
__global__ void buildBoxesKernel(cuBQL::box3f* aabbs, const float* samples, int num_samples) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_samples) return;

    float x = samples[3*idx + 0];
    float y = samples[3*idx + 1];
    float z = samples[3*idx + 2];
    aabbs[idx] = cuBQL::box3f(cuBQL::vec3f(x, y, z));
}

class GaussianModel {
private:
    int num_gaussians;
    
    // Device pointers for Gaussian parameters
    float* d_means;      // xyz coordinates (3 * num_gaussians)
    float* d_scales;     // scale parameters (3 * num_gaussians) 
    float* d_rotations;  // rotation quaternions (4 * num_gaussians)
    float* d_values;     // value parameters (num_gaussians)
    float* d_weights;    // weight parameters (num_gaussians)
    
    bool data_loaded;
    
    // BVH and temporary data for inference
    cuBQL::bvh3f samples_bvh;
    cuBQL::bvh3f gaussian_bvh;
    float* d_stored_samples;   // Copy of samples for GPU processing
    int num_stored_samples;

public:
    GaussianModel() : num_gaussians(0), d_means(nullptr), d_scales(nullptr), 
                      d_rotations(nullptr), d_values(nullptr), d_weights(nullptr), 
                      data_loaded(false), d_stored_samples(nullptr), num_stored_samples(0) {
        // Initialize BVH structures
        samples_bvh = cuBQL::bvh3f();
        gaussian_bvh = cuBQL::bvh3f();
    }
    
    ~GaussianModel() {
        cleanup();
    }
    
    void cleanup() {
        if (d_means) { cudaFree(d_means); d_means = nullptr; }
        if (d_scales) { cudaFree(d_scales); d_scales = nullptr; }
        if (d_rotations) { cudaFree(d_rotations); d_rotations = nullptr; }
        if (d_values) { cudaFree(d_values); d_values = nullptr; }
        if (d_weights) { cudaFree(d_weights); d_weights = nullptr; }
        if (d_stored_samples) { cudaFree(d_stored_samples); d_stored_samples = nullptr; }
        
        // Clean up BVH structures
        cuBQL::cuda::free(samples_bvh);
        samples_bvh.nodes    = nullptr;
        samples_bvh.primIDs  = nullptr;
        samples_bvh.numNodes = 0;
        samples_bvh.numPrims = 0;
        cuBQL::cuda::free(gaussian_bvh);
        gaussian_bvh.nodes    = nullptr;
        gaussian_bvh.primIDs  = nullptr;
        gaussian_bvh.numNodes = 0;
        gaussian_bvh.numPrims = 0;
        data_loaded = false;
        num_stored_samples = 0;
    }
    
    bool loadPly(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open PLY file " << filename << std::endl;
            return false;
        }
        
        // Parse header
        std::string line;
        bool is_binary = false;
        bool is_little_endian = true;
        int vertex_count = 0;
        
        while (std::getline(file, line)) {
            if (line == "end_header") break;
            
            if (line.substr(0, 6) == "format") {
                if (line.find("binary_little_endian") != std::string::npos) {
                    is_binary = true;
                    is_little_endian = true;
                } else if (line.find("binary_big_endian") != std::string::npos) {
                    is_binary = true;
                    is_little_endian = false;
                } else {
                    std::cerr << "Error: Only binary PLY format supported" << std::endl;
                    return false;
                }
            }
            
            if (line.substr(0, 14) == "element vertex") {
                std::stringstream ss(line);
                std::string element, vertex;
                ss >> element >> vertex >> vertex_count;
            }
        }
        
        if (!is_binary || vertex_count <= 0) {
            std::cerr << "Error: Invalid PLY format or no vertices" << std::endl;
            return false;
        }
        
        cleanup();
        num_gaussians = vertex_count;
        
        // Allocate host memory for reading
        std::vector<float> h_means(3 * num_gaussians);
        std::vector<float> h_scales(3 * num_gaussians); 
        std::vector<float> h_rotations(4 * num_gaussians);
        std::vector<float> h_values(num_gaussians);
        std::vector<float> h_weights(num_gaussians);
        
        // Read binary data (order: x,y,z,value,weight,scale_0,scale_1,scale_2,rot_0,rot_1,rot_2,rot_3)
        for (int i = 0; i < num_gaussians; i++) {
            float data[12];
            file.read(reinterpret_cast<char*>(data), 12 * sizeof(float));
            
            if (!file.good()) {
                std::cerr << "Error: Failed to read vertex data at index " << i << std::endl;
                return false;
            }
            
            // Handle big endian if needed
            if (!is_little_endian) {
                for (int j = 0; j < 12; j++) {
                    char* bytes = reinterpret_cast<char*>(&data[j]);
                    std::swap(bytes[0], bytes[3]);
                    std::swap(bytes[1], bytes[2]);
                }
            }
            
            // Store in interleaved format
            h_means[3*i + 0] = data[0];    // x
            h_means[3*i + 1] = data[1];    // y  
            h_means[3*i + 2] = data[2];    // z
            h_values[i] = data[3];         // value
            h_weights[i] = data[4];        // weight
            h_scales[3*i + 0] = data[5];   // scale_0
            h_scales[3*i + 1] = data[6];   // scale_1
            h_scales[3*i + 2] = data[7];   // scale_2
            h_rotations[4*i + 0] = data[8];  // rot_0
            h_rotations[4*i + 1] = data[9];  // rot_1
            h_rotations[4*i + 2] = data[10]; // rot_2
            h_rotations[4*i + 3] = data[11]; // rot_3
        }
        
        file.close();
        
        // Allocate device memory
        CHECK_CUDA(cudaMalloc((void**)&d_means, 3 * num_gaussians * sizeof(float)), true);
        CHECK_CUDA(cudaMalloc((void**)&d_scales, 3 * num_gaussians * sizeof(float)), true);
        CHECK_CUDA(cudaMalloc((void**)&d_rotations, 4 * num_gaussians * sizeof(float)), true);
        CHECK_CUDA(cudaMalloc((void**)&d_values, num_gaussians * sizeof(float)), true);
        CHECK_CUDA(cudaMalloc((void**)&d_weights, num_gaussians * sizeof(float)), true);
        
        // Copy data to device
        CHECK_CUDA(cudaMemcpy(d_means, h_means.data(), 3 * num_gaussians * sizeof(float), cudaMemcpyHostToDevice), true);
        CHECK_CUDA(cudaMemcpy(d_scales, h_scales.data(), 3 * num_gaussians * sizeof(float), cudaMemcpyHostToDevice), true);
        CHECK_CUDA(cudaMemcpy(d_rotations, h_rotations.data(), 4 * num_gaussians * sizeof(float), cudaMemcpyHostToDevice), true);
        CHECK_CUDA(cudaMemcpy(d_values, h_values.data(), num_gaussians * sizeof(float), cudaMemcpyHostToDevice), true);
        CHECK_CUDA(cudaMemcpy(d_weights, h_weights.data(), num_gaussians * sizeof(float), cudaMemcpyHostToDevice), true);

        data_loaded = true;
        std::cout << "Successfully loaded " << num_gaussians << " Gaussians from " << filename << std::endl;
        return true;
    }
    
    // Getter methods
    float* getMeans() const { return d_means; }
    float* getScales() const { return d_scales; }
    float* getRotations() const { return d_rotations; }
    float* getValues() const { return d_values; }
    float* getWeights() const { return d_weights; }
    int getNumGaussians() const { return num_gaussians; }
    bool isDataLoaded() const { return data_loaded; }
    
    // Modified inference method that works with GPU buffers directly
    bool infer(const float* d_samples, int num_samples, float* d_out_values, float* d_out_weights,
            float scale_modifier = 1.0f, float background = 0.0f, 
            bool use_gaussian_bvh = false, bool debug = false) {
        
        if (!data_loaded || !d_samples || !d_out_values || !d_out_weights || num_samples <= 0) {
            std::cerr << "Error: Invalid input parameters for inference" << std::endl;
            return false;
        }
        
        if (debug) {
            std::cout << "Starting inference with " << num_samples << " samples and " 
                    << num_gaussians << " Gaussians" << std::endl;
        }
        
        // Store reference to samples buffer (no copying needed since already on GPU)
        if (d_stored_samples && num_stored_samples != num_samples) {
            cudaFree(d_stored_samples);
            d_stored_samples = nullptr;
        }
        
        // Use the provided GPU buffer directly
        d_stored_samples = const_cast<float*>(d_samples);
        num_stored_samples = num_samples;
        
        // Build BVH for samples
        buildSamplesBVH(debug);
        
        // Allocate device memory for conics (internal use only)
        float* d_conics;
        CHECK_CUDA(cudaMalloc((void**)&d_conics, num_gaussians * 6 * sizeof(float)), debug);
        
        // Initialize outputs (caller-provided buffers)
        CHECK_CUDA(cudaMemset(d_out_values, 0, num_samples * sizeof(float)), debug);
        CHECK_CUDA(cudaMemset(d_out_weights, 0, num_samples * sizeof(float)), debug);

        // Free Gaussian bvh
        cuBQL::cuda::free(gaussian_bvh);
        gaussian_bvh.nodes    = nullptr;
        gaussian_bvh.primIDs  = nullptr;
        gaussian_bvh.numNodes = 0;
        gaussian_bvh.numPrims = 0;
        gaussian_bvh = cuBQL::bvh3f();
        
        try {
            // Call the existing CUDA rasterizer forward function
            CudaRasterizer::Rasterizer::forward(
                num_gaussians,          // P - number of Gaussians
                num_samples,            // S - number of samples  
                d_means,                // means3D
                d_scales,               // scales
                scale_modifier,         // scale_modifier
                d_rotations,            // rotations
                d_values,               // values
                d_weights,              // weights
                d_stored_samples,       // samples (GPU buffer)
                samples_bvh,            // BVH for samples
                gaussian_bvh,           // BVH for Gaussians (will be built if use_gaussian_bvh)
                d_conics,               // output conics
                d_out_values,           // output values (caller's GPU buffer)
                d_out_weights,          // output weights (caller's GPU buffer)
                use_gaussian_bvh,       // whether to use Gaussian BVH
                debug                   // debug flag
            );
            
        } catch (const std::exception& e) {
            std::cerr << "Error during rasterization: " << e.what() << std::endl;
            cudaFree(d_conics);
            d_stored_samples = nullptr; // Don't free since we don't own it
            return false;
        }
        
        // Clean up temporary device memory
        cudaFree(d_conics);
        d_stored_samples = nullptr; // Don't free since we don't own it

        // CHECK_CUDA(cudaMemset(d_out_values, 0.5, num_samples * sizeof(float)), debug);
        
        return true;
    }

private:
    
    // Build BVH for samples
    void buildSamplesBVH(bool debug = false) {
        if (num_stored_samples <= 0) return;
        
        cudaEvent_t gpuStart, gpuStop;
        if (debug) {
            cudaEventCreate(&gpuStart);
            cudaEventCreate(&gpuStop);
            cudaEventRecord(gpuStart, 0);
        }
        
        // Free existing BVH
        cuBQL::cuda::free(samples_bvh);
        samples_bvh.nodes    = nullptr;
        samples_bvh.primIDs  = nullptr;
        samples_bvh.numNodes = 0;
        samples_bvh.numPrims = 0;
        samples_bvh = cuBQL::bvh3f();
        
        // Allocate bounding boxes
        cuBQL::box3f* d_boxes;
        cudaMalloc(&d_boxes, num_stored_samples * sizeof(cuBQL::box3f));

        const int threads = 256;
        const int blocks = (num_stored_samples + threads - 1) / threads;
        buildBoxesKernel<<<blocks, threads>>>(d_boxes, d_stored_samples, num_stored_samples);
        
        // Build BVH using cuBQL
        cuBQL::BuildConfig cfg;
        cfg.makeLeafThreshold = 33;
        cuBQL::cuda::radixBuilder(samples_bvh, d_boxes, num_stored_samples, cfg);
        
        cudaFree(d_boxes);

        if (debug) {
            cudaEventRecord(gpuStop, 0);
            cudaEventSynchronize(gpuStop);  
            float msBoxes = 0.f;
            cudaEventElapsedTime(&msBoxes, gpuStart, gpuStop);
            std::cout << "Sample BVH time: " << msBoxes << " ms" << std::endl;
        }
    }
};