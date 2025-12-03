#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <cassert>

#include "gaussian_model.h"

class GaussianModelTester {
private:
    GaussianModel model;
    std::vector<float> test_samples;
    std::vector<float> output_values;
    std::vector<float> output_weights;
    int num_test_points;
    
public:
    GaussianModelTester(int num_points = 100) : num_test_points(num_points) {
        generateRandomSamples();
        output_values.resize(num_test_points);
        output_weights.resize(num_test_points);
    }
    
    void generateRandomSamples() {
        test_samples.clear();
        test_samples.reserve(3 * num_test_points);
        
        // Use a fixed seed for reproducible results
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        
        std::cout << "Generating " << num_test_points << " random test points in [0,1]^3..." << std::endl;
        
        for (int i = 0; i < num_test_points; i++) {
            float x = dist(gen);
            float y = dist(gen);
            float z = dist(gen);
            
            test_samples.push_back(x);
            test_samples.push_back(y);
            test_samples.push_back(z);
            
            // Print first few samples for verification
            if (i < 5) {
                std::cout << "Sample " << i << ": (" << x << ", " << y << ", " << z << ")" << std::endl;
            }
        }
        std::cout << "Sample generation completed." << std::endl;
    }
    
    bool loadModel(const std::string& ply_file) {
        std::cout << "\n=== Loading Gaussian Model ===" << std::endl;
        std::cout << "Loading PLY file: " << ply_file << std::endl;
        
        auto start = std::chrono::high_resolution_clock::now();
        bool success = model.loadPly(ply_file);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        if (success) {
            std::cout << "Model loaded successfully in " << duration.count() << " ms" << std::endl;
            std::cout << "Number of Gaussians: " << model.getNumGaussians() << std::endl;
        } else {
            std::cerr << "Failed to load model from " << ply_file << std::endl;
        }
        
        return success;
    }
    
    // Updated test function that manages GPU buffers
    bool runInferenceTest(bool use_gaussian_bvh = false, float scale_modifier = 1.0f, 
                        float background = 0.0f, bool debug = false) {
        std::cout << "\n=== Running Inference Test ===" << std::endl;
        std::cout << "Test parameters:" << std::endl;
        std::cout << "  - Number of test points: " << num_test_points << std::endl;
        std::cout << "  - Use Gaussian BVH: " << (use_gaussian_bvh ? "Yes" : "No") << std::endl;
        std::cout << "  - Scale modifier: " << scale_modifier << std::endl;
        std::cout << "  - Background: " << background << std::endl;
        std::cout << "  - Debug mode: " << (debug ? "Yes" : "No") << std::endl;
        
        // Allocate GPU buffers for samples and outputs
        float* d_test_samples;
        float* d_output_values;
        float* d_output_weights;
        
        CHECK_CUDA(cudaMalloc((void**)&d_test_samples, 3 * num_test_points * sizeof(float)), debug);
        CHECK_CUDA(cudaMalloc((void**)&d_output_values, num_test_points * sizeof(float)), debug);
        CHECK_CUDA(cudaMalloc((void**)&d_output_weights, num_test_points * sizeof(float)), debug);
        
        // Copy test samples to GPU
        CHECK_CUDA(cudaMemcpy(d_test_samples, test_samples.data(), 
                            3 * num_test_points * sizeof(float), cudaMemcpyHostToDevice), debug);
        
        // Clear host output arrays
        std::fill(output_values.begin(), output_values.end(), 0.0f);
        std::fill(output_weights.begin(), output_weights.end(), 0.0f);
        
        auto start = std::chrono::high_resolution_clock::now();
        
        bool success = model.infer(
            d_test_samples,      // samples (GPU buffer)
            num_test_points,     // num_samples
            d_output_values,     // out_values (GPU buffer)
            d_output_weights,    // out_weights (GPU buffer)
            scale_modifier,      // scale_modifier
            background,          // background
            use_gaussian_bvh,    // use_gaussian_bvh
            debug                // debug
        );
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        if (success) {
            // Copy results back to host
            CHECK_CUDA(cudaMemcpy(output_values.data(), d_output_values, 
                                num_test_points * sizeof(float), cudaMemcpyDeviceToHost), debug);
            CHECK_CUDA(cudaMemcpy(output_weights.data(), d_output_weights, 
                                num_test_points * sizeof(float), cudaMemcpyDeviceToHost), debug);
            
            std::cout << "Inference completed successfully in " << duration.count() << " μs" << std::endl;
            std::cout << "Average time per sample: " << (double)duration.count() / num_test_points << " μs" << std::endl;
        } else {
            std::cerr << "Inference failed!" << std::endl;
        }
        
        // Clean up GPU buffers
        cudaFree(d_test_samples);
        cudaFree(d_output_values);
        cudaFree(d_output_weights);
        
        return success;
    }
    
    void analyzeResults() {
        std::cout << "\n=== Results Analysis ===" << std::endl;
        
        // Calculate statistics
        float min_value = *std::min_element(output_values.begin(), output_values.end());
        float max_value = *std::max_element(output_values.begin(), output_values.end());
        float min_weight = *std::min_element(output_weights.begin(), output_weights.end());
        float max_weight = *std::max_element(output_weights.begin(), output_weights.end());
        
        float sum_values = 0.0f, sum_weights = 0.0f;
        for (int i = 0; i < num_test_points; i++) {
            sum_values += output_values[i];
            sum_weights += output_weights[i];
        }
        float avg_value = sum_values / num_test_points;
        float avg_weight = sum_weights / num_test_points;
        
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Output Values Statistics:" << std::endl;
        std::cout << "  Min: " << min_value << std::endl;
        std::cout << "  Max: " << max_value << std::endl;
        std::cout << "  Avg: " << avg_value << std::endl;
        
        std::cout << "Output Weights Statistics:" << std::endl;
        std::cout << "  Min: " << min_weight << std::endl;
        std::cout << "  Max: " << max_weight << std::endl;
        std::cout << "  Avg: " << avg_weight << std::endl;
        
        // Show first 10 results
        std::cout << "\nFirst 10 sample results:" << std::endl;
        std::cout << "Index | Sample Point (x, y, z)        | Value      | Weight" << std::endl;
        std::cout << "------|------------------------------|------------|------------" << std::endl;
        for (int i = 0; i < std::min(10, num_test_points); i++) {
            float x = test_samples[3*i + 0];
            float y = test_samples[3*i + 1];
            float z = test_samples[3*i + 2];
            
            std::cout << std::setw(5) << i << " | "
                      << "(" << std::setw(8) << x << ", " 
                      << std::setw(8) << y << ", " 
                      << std::setw(8) << z << ") | "
                      << std::setw(10) << output_values[i] << " | "
                      << std::setw(10) << output_weights[i] << std::endl;
        }
        
        // Check for any NaN or infinite values
        int nan_values = 0, nan_weights = 0;
        int inf_values = 0, inf_weights = 0;
        
        for (int i = 0; i < num_test_points; i++) {
            if (std::isnan(output_values[i])) nan_values++;
            if (std::isnan(output_weights[i])) nan_weights++;
            if (std::isinf(output_values[i])) inf_values++;
            if (std::isinf(output_weights[i])) inf_weights++;
        }
        
        if (nan_values > 0 || nan_weights > 0 || inf_values > 0 || inf_weights > 0) {
            std::cout << "\nWarning: Invalid values detected:" << std::endl;
            if (nan_values > 0) std::cout << "  NaN values: " << nan_values << std::endl;
            if (nan_weights > 0) std::cout << "  NaN weights: " << nan_weights << std::endl;
            if (inf_values > 0) std::cout << "  Infinite values: " << inf_values << std::endl;
            if (inf_weights > 0) std::cout << "  Infinite weights: " << inf_weights << std::endl;
        } else {
            std::cout << "\nAll output values are valid (no NaN or infinite values)" << std::endl;
        }
    }
    
    void runFullTest(const std::string& ply_file) {
        std::cout << "Starting GaussianModel Test Suite" << std::endl;
        std::cout << "=================================" << std::endl;
        
        // Test 1: Load model
        if (!loadModel(ply_file)) {
            std::cerr << "Cannot proceed with tests - model loading failed" << std::endl;
            return;
        }
        
        // Test 2: Basic inference without BVH
        std::cout << "\n[Test 1] Basic inference without Gaussian BVH" << std::endl;
        if (runInferenceTest(false, 1.0f, 0.0f, true)) {
            analyzeResults();
        }
        
        // // Test 3: Inference with Gaussian BVH
        // std::cout << "\n[Test 2] Inference with Gaussian BVH" << std::endl;
        // if (runInferenceTest(true, 1.0f, 0.0f, true)) {
        //     analyzeResults();
        // }
        
        // // Test 4: Different scale modifier
        // std::cout << "\n[Test 3] Inference with modified scale (2.0x)" << std::endl;
        // if (runInferenceTest(false, 2.0f, 0.0f, false)) {
        //     analyzeResults();
        // }
        
        std::cout << "\n=== Test Suite Complete ===" << std::endl;
    }
    
    void saveResultsToFile(const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open output file: " << filename << std::endl;
            return;
        }
        
        file << "# GaussianModel Inference Results\n";
        file << "# Format: sample_x sample_y sample_z output_value output_weight\n";
        
        for (int i = 0; i < num_test_points; i++) {
            file << test_samples[3*i + 0] << " "
                 << test_samples[3*i + 1] << " "
                 << test_samples[3*i + 2] << " "
                 << output_values[i] << " "
                 << output_weights[i] << "\n";
        }
        
        file.close();
        std::cout << "Results saved to " << filename << std::endl;
    }
};

int main(int argc, char* argv[]) {
    // Check CUDA availability
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    
    if (error != cudaSuccess || device_count == 0) {
        std::cerr << "No CUDA devices found or CUDA not available!" << std::endl;
        return EXIT_FAILURE;
    }
    
    std::cout << "Found " << device_count << " CUDA device(s)" << std::endl;
    
    // Set device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Using device: " << prop.name << std::endl;
    
    // Determine PLY file path
    std::string ply_file = "point_cloud.ply";
    if (argc > 1) {
        ply_file = argv[1];
    }
    
    // Determine number of test points
    int num_points = 100;
    if (argc > 2) {
        num_points = std::atoi(argv[2]);
        if (num_points <= 0) {
            std::cerr << "Invalid number of points: " << argv[2] << std::endl;
            return EXIT_FAILURE;
        }
    }
    
    std::cout << "Test parameters:" << std::endl;
    std::cout << "  PLY file: " << ply_file << std::endl;
    std::cout << "  Number of test points: " << num_points << std::endl;
    std::cout << std::endl;
    
    try {
        GaussianModelTester tester(num_points);
        tester.runFullTest(ply_file);
        
        // Save results to file
        tester.saveResultsToFile("inference_results.txt");
        
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    
    std::cout << "Test completed successfully!" << std::endl;
    return EXIT_SUCCESS;
}