#define R 1.0
#define GAMMA 1.4
#define CV (R/(GAMMA-1.0))
#define VALUES_PER_CELL 3

__kernel void compute_p_from_u(__global const float *u,
                               __global float *p,
                               const int n) {
    // Get the global cell ID of the current work item
    // Here we assume that each cell has one OpenCL thread working on it.
    int cell_id = get_global_id(0);
    // Make sure we don't go out of bounds
    if (cell_id < n) {
        // Density is an easy mapping - do it below.
        // X Velocity = X Mom / Density
        float xvel = u[(cell_id*VALUES_PER_CELL)+1]/u[(cell_id*VALUES_PER_CELL)];
        // Energy is a bit more complex
        float specific_eng = u[(cell_id*VALUES_PER_CELL)+2]/u[(cell_id*VALUES_PER_CELL)];
        // Subtract away v2 and divide by Cv
        specific_eng = specific_eng - 0.5*xvel*xvel;

        p[cell_id*VALUES_PER_CELL] = u[cell_id*VALUES_PER_CELL];
        p[(cell_id*VALUES_PER_CELL)+1] = xvel;
        p[(cell_id*VALUES_PER_CELL)+2] = specific_eng/CV;
    }
} 