#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <CL/cl.h>

#define MAX_SOURCE_SIZE (0x100000)
#define R 1.0
#define GAMMA 1.4
#define CV (R/(GAMMA-1.0))
#define VALUES_PER_CELL 3

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

    /*
    ============ OpenCL preparations ==============
    */

    cl_int ret;
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

    /*
    ============ SHLL source and program preparations ==============
    */

    // Read kernel source for p_from_u computation
    char* kernel_source = read_kernel_source("shll_p_from_u.cl");
    
    // Create a program from the kernel source for p_from_u computation
    cl_program p_from_u_program = clCreateProgramWithSource(context, 1, (const char**)&kernel_source, NULL, &ret);
    check_error(ret, "clCreateProgramWithSource");
    
    // Build the program
    ret = clBuildProgram(p_from_u_program, 1, &device_id, NULL, NULL, NULL);
    if (ret != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(p_from_u_program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = malloc(log_size);
        clGetProgramBuildInfo(p_from_u_program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        fprintf(stderr, "Program build failed:\n%s\n", log);
        free(log);
        exit(1);
    }

    // Now for split flux computation (f from p)
    kernel_source = read_kernel_source("shll_f_from_p.cl");
    cl_program f_from_p_program = clCreateProgramWithSource(context, 1, (const char**)&kernel_source, NULL, &ret);
    check_error(ret, "clCreateProgramWithSource");
    
    // Build the program
    ret = clBuildProgram(f_from_p_program, 1, &device_id, NULL, NULL, NULL);
    if (ret != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(f_from_p_program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = malloc(log_size);
        clGetProgramBuildInfo(f_from_p_program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        fprintf(stderr, "Program build failed:\n%s\n", log);
        free(log);
        exit(1);
    }
    /*
    ============ SHLL solver preparations ==============
    */

    // Now we allocate the memory for the SHLL solver.
    const int N = 1000;
    const int NO_STEPS = 1;
    float *p = (float*)malloc(N * VALUES_PER_CELL * sizeof(float));
    float *u = (float*)malloc(N * VALUES_PER_CELL * sizeof(float));
    float *fp = (float*)malloc(N * VALUES_PER_CELL * sizeof(float));
    float *fm = (float*)malloc(N * VALUES_PER_CELL * sizeof(float));

    // Set the values of U first
    for (int i = 0; i < N; i++) {
        u[i*VALUES_PER_CELL] = 1.0; // Density
        u[i*VALUES_PER_CELL+1] = 1.0*0.5; // Momentum (speed = 0.5)
        u[i*VALUES_PER_CELL+2] = 1.0*(CV*1.0 + 0.5*0.5*0.5); // Energy (temp = 1.0)
    }
    
    // Create memory buffers on the device
    cl_mem p_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, N * VALUES_PER_CELL * sizeof(float), NULL, &ret);
    check_error(ret, "clCreateBuffer for p");
    cl_mem u_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, N * VALUES_PER_CELL * sizeof(float), NULL, &ret);
    check_error(ret, "clCreateBuffer for u");
    cl_mem fp_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, N * VALUES_PER_CELL * sizeof(float), NULL, &ret);
    check_error(ret, "clCreateBuffer for fp");
    cl_mem fm_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, N * VALUES_PER_CELL * sizeof(float), NULL, &ret);
    check_error(ret, "clCreateBuffer for fm");

    // Copy the data for u from the host into its memory buffer
    ret = clEnqueueWriteBuffer(command_queue, u_mem_obj, CL_TRUE, 0, N * VALUES_PER_CELL * sizeof(float), u, 0, NULL, NULL);
    check_error(ret, "clEnqueueWriteBuffer for p");

    /*
    -------- SHLL Compute P from U Kernel Definition ----------
    */
    // Create the OpenCL kernel for computing P from U
    cl_kernel p_from_u_kernel = clCreateKernel(p_from_u_program, "compute_p_from_u", &ret);
    check_error(ret, "clCreateKernel");
    // Set the arguments of the p from u kernel
    ret = clSetKernelArg(p_from_u_kernel, 0, sizeof(cl_mem), (void*)&u_mem_obj);
    check_error(ret, "clSetKernelArg 0");
    ret = clSetKernelArg(p_from_u_kernel, 1, sizeof(cl_mem), (void*)&p_mem_obj);
    check_error(ret, "clSetKernelArg 1");
    ret = clSetKernelArg(p_from_u_kernel, 2, sizeof(int), (void*)&N);
    check_error(ret, "clSetKernelArg 2");

    /*
    -------- SHLL Compute F from P Kernel Definition ----------
    */
    cl_kernel f_from_p_kernel = clCreateKernel(f_from_p_program, "compute_f_from_p", &ret);
    check_error(ret, "clCreateKernel");
    // Set the arguments of the p from u kernel
    ret = clSetKernelArg(f_from_p_kernel, 0, sizeof(cl_mem), (void*)&p_mem_obj);
    check_error(ret, "clSetKernelArg 0");
    ret = clSetKernelArg(f_from_p_kernel, 1, sizeof(cl_mem), (void*)&u_mem_obj);
    check_error(ret, "clSetKernelArg 0");
    ret = clSetKernelArg(f_from_p_kernel, 2, sizeof(cl_mem), (void*)&fp_mem_obj);
    check_error(ret, "clSetKernelArg 1");
    ret = clSetKernelArg(f_from_p_kernel, 3, sizeof(cl_mem), (void*)&fm_mem_obj);
    check_error(ret, "clSetKernelArg 1");
    ret = clSetKernelArg(f_from_p_kernel, 4, sizeof(int), (void*)&N);
    check_error(ret, "clSetKernelArg 2");


    // Execute the OpenCL kernel on the list
    size_t global_item_size = N; // Process the entire list
    size_t local_item_size = get_optimal_local_size(device_id, global_item_size);
    
    // Try to execute with local work group size, fallback to NULL if it fails
    ret = clEnqueueNDRangeKernel(command_queue, p_from_u_kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);
    check_error(ret, "clEnqueueNDRangeKernel (P-from-U State Computation)");

    // Start time stepping
    for (int step = 0; step < NO_STEPS; step++) {
        printf("Step %d of %d\n", step, NO_STEPS);
        // Compute the split fluxes
        ret = clEnqueueNDRangeKernel(command_queue, f_from_p_kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);
        check_error(ret, "clEnqueueNDRangeKernel (Flux Computation)");
    }

    // Read the memory buffer fm on the device to the local variable p
    ret = clEnqueueReadBuffer(command_queue, p_mem_obj, CL_TRUE, 0, N * VALUES_PER_CELL * sizeof(float), p, 0, NULL, NULL);
    check_error(ret, "clEnqueueReadBuffer");
    
    // Display the result
    printf("State Computation Results (Euler Equations):\n");
    printf("First 10 elements:\n");
    for (int i = 0; i < 10; i++) {
        printf("Cell [%d] state = %.2f, %.2f, %.2f\n", i, p[VALUES_PER_CELL*i], p[VALUES_PER_CELL*i+1], p[VALUES_PER_CELL*i+2]);
    }
   
    // Clean up
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    // P from U kernel (Euler equation state computation)
    ret = clReleaseKernel(p_from_u_kernel);
    ret = clReleaseProgram(p_from_u_program);
    // F from P kernel (SHLL split flux computation)
    ret = clReleaseKernel(f_from_p_kernel);
    ret = clReleaseProgram(f_from_p_program);
    ret = clReleaseMemObject(p_mem_obj);
    ret = clReleaseMemObject(u_mem_obj);
    ret = clReleaseMemObject(fp_mem_obj);
    ret = clReleaseMemObject(fm_mem_obj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
    
    free(p);
    free(u);
    free(fp);
    free(fm);
    free(kernel_source);
    
    printf("SHLL solver completed successfully!\n");
    return 0;
} 