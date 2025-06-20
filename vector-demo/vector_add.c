#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <CL/cl.h>

#define MAX_SOURCE_SIZE (0x100000)

// Function to check OpenCL errors
void check_error(cl_int error, const char* operation) {
    if (error != CL_SUCCESS) {
        fprintf(stderr, "Error during %s: %d\n", operation, error);
        exit(1);
    }
}

// Function to read kernel source from file
char* read_kernel_source(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Failed to open kernel file: %s\n", filename);
        exit(1);
    }
    
    char* source = malloc(MAX_SOURCE_SIZE);
    size_t source_size = fread(source, 1, MAX_SOURCE_SIZE, file);
    fclose(file);
    
    return source;
}

// Function to get optimal local work group size
size_t get_optimal_local_size(cl_device_id device_id, size_t global_size) {
    // Get device name
    char device_name[256];
    cl_int ret = clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
    if (ret == CL_SUCCESS) {
        printf("Using OpenCL device: %s\n", device_name);
    } else {
        printf("Using OpenCL device: (unknown)\n");
    }
    
    size_t max_work_group_size;
    ret = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, 
                         sizeof(size_t), &max_work_group_size, NULL);
    if (ret != CL_SUCCESS) {
        fprintf(stderr, "Warning: Could not get max work group size, using 1\n");
        return 1;
    }
    
    // Use the minimum of max work group size and global size
    size_t local_size = (max_work_group_size < global_size) ? max_work_group_size : global_size;
    
    // Make sure local_size divides global_size evenly
    while (global_size % local_size != 0 && local_size > 1) {
        local_size--;
    }
    
    printf("Using local work group size: %zu (max: %zu)\n", local_size, max_work_group_size);
    return local_size;
}

int main() {
    cl_int ret;
    const int N = 100;
    // Get platform and device information
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    
    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    check_error(ret, "clGetPlatformIDs");
    
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);
    check_error(ret, "clGetDeviceIDs");
    
    // Create an OpenCL context
    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    check_error(ret, "clCreateContext");
    
    // Create a command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
    check_error(ret, "clCreateCommandQueue");
    
    // Read kernel source
    char* kernel_source = read_kernel_source("vector_add.cl");
    
    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernel_source, NULL, &ret);
    check_error(ret, "clCreateProgramWithSource");
    
    // Build the program
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (ret != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = malloc(log_size);
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        fprintf(stderr, "Program build failed:\n%s\n", log);
        free(log);
        exit(1);
    }
    
    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "vector_add", &ret);
    check_error(ret, "clCreateKernel");
    
    // Initialize input vectors
    float* a = (float*)malloc(N * sizeof(float));
    float* b = (float*)malloc(N * sizeof(float));
    float* c = (float*)malloc(N * sizeof(float));
    
    for (int i = 0; i < N; i++) {
        a[i] = (float)i;
        b[i] = (float)(i * 2);
        c[i] = 0.0f;
    }
    
    // Create memory buffers on the device
    cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, N * sizeof(float), NULL, &ret);
    check_error(ret, "clCreateBuffer for a");
    
    cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, N * sizeof(float), NULL, &ret);
    check_error(ret, "clCreateBuffer for b");
    
    cl_mem c_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, N * sizeof(float), NULL, &ret);
    check_error(ret, "clCreateBuffer for c");
    
    // Copy the lists a and b to their respective memory buffers
    ret = clEnqueueWriteBuffer(command_queue, a_mem_obj, CL_TRUE, 0, N * sizeof(float), a, 0, NULL, NULL);
    check_error(ret, "clEnqueueWriteBuffer for a");
    
    ret = clEnqueueWriteBuffer(command_queue, b_mem_obj, CL_TRUE, 0, N * sizeof(float), b, 0, NULL, NULL);
    check_error(ret, "clEnqueueWriteBuffer for b");
    
    // Set the arguments of the kernel
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&a_mem_obj);
    check_error(ret, "clSetKernelArg 0");
    
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&b_mem_obj);
    check_error(ret, "clSetKernelArg 1");
    
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&c_mem_obj);
    check_error(ret, "clSetKernelArg 2");
    
    ret = clSetKernelArg(kernel, 3, sizeof(int), (void*)&N);
    check_error(ret, "clSetKernelArg 3");
    
    // Execute the OpenCL kernel on the list
    size_t global_item_size = N; // Process the entire list
    size_t local_item_size = get_optimal_local_size(device_id, global_item_size);
    
    // Try to execute with local work group size, fallback to NULL if it fails
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);
    if (ret == CL_INVALID_WORK_GROUP_SIZE) {
        printf("Falling back to automatic local work group size\n");
        ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_item_size, NULL, 0, NULL, NULL);
    }
    check_error(ret, "clEnqueueNDRangeKernel");
    
    // Read the memory buffer c on the device to the local variable c
    ret = clEnqueueReadBuffer(command_queue, c_mem_obj, CL_TRUE, 0, N * sizeof(float), c, 0, NULL, NULL);
    check_error(ret, "clEnqueueReadBuffer");
    
    // Display the result
    printf("Vector Addition Results (showing first 10 elements):\n");
    for (int i = 0; i < 10; i++) {
        printf("c[%d] = %.2f + %.2f = %.2f\n", i, a[i], b[i], c[i]);
    }
    printf("...\n");
    printf("c[%d] = %.2f + %.2f = %.2f\n", N-1, a[N-1], b[N-1], c[N-1]);
    
    // Clean up
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(a_mem_obj);
    ret = clReleaseMemObject(b_mem_obj);
    ret = clReleaseMemObject(c_mem_obj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
    
    free(a);
    free(b);
    free(c);
    free(kernel_source);
    
    printf("Vector addition completed successfully!\n");
    return 0;
} 