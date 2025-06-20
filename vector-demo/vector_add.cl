__kernel void vector_add(__global const float* a,
                        __global const float* b,
                        __global float* c,
                        const int n) {
    // Get the global ID of the current work item
    int gid = get_global_id(0);
    
    // Make sure we don't go out of bounds
    if (gid < n) {
        c[gid] = a[gid] + b[gid];
    }
} 