#define VALUES_PER_CELL 3
#define DT 0.0001
#define L 1.0


__kernel void compute_u_from_f(__global const float *p,
                               __global const float *fp,
                               __global const float *fm,
                               __global float *u,
                               const int n) {
    // Get the global cell ID of the current work item
    // Here we assume that each cell has one OpenCL thread working on it.
    int cell_id = get_global_id(0);
    // Make sure we don't go out of bounds
    if (cell_id < n) {

        // Update state based on fluxes
        float FR[3];
        float FL[3];
        float DX = L/n;
        int left_index, right_index;
        // Use outflow bounds; its easier for testing
        if (cell_id == 0) {
            left_index = cell_id;
            right_index = cell_id + 1;
        } else if (cell_id == (n-1)) {
            left_index = cell_id - 1;
            right_index = cell_id;
        } else {
            left_index = cell_id - 1;
            right_index = cell_id + 1;
        }

        FL[0] = fp[left_index*VALUES_PER_CELL] + fm[cell_id*VALUES_PER_CELL];
        FL[1] = fp[left_index*VALUES_PER_CELL+1] + fm[cell_id*VALUES_PER_CELL+1];
        FL[2] = fp[left_index*VALUES_PER_CELL+2] + fm[cell_id*VALUES_PER_CELL+2];

        FR[0] = fp[cell_id*VALUES_PER_CELL] + fm[right_index*VALUES_PER_CELL];
        FR[1] = fp[cell_id*VALUES_PER_CELL+1] + fm[right_index*VALUES_PER_CELL+1];
        FR[2] = fp[cell_id*VALUES_PER_CELL+2] + fm[right_index*VALUES_PER_CELL+2];

        u[cell_id*VALUES_PER_CELL + 0] = u[cell_id*VALUES_PER_CELL + 0] - (DT/DX)*(FR[0] - FL[0]);
        u[cell_id*VALUES_PER_CELL + 1] = u[cell_id*VALUES_PER_CELL + 1] - (DT/DX)*(FR[1] - FL[1]);
        u[cell_id*VALUES_PER_CELL + 2] = u[cell_id*VALUES_PER_CELL + 2] - (DT/DX)*(FR[2] - FL[2]);
    }
}