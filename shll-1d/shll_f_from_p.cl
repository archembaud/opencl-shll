#define R 1.0
#define GAMMA 1.4
#define CV (R/(GAMMA-1.0))
#define CP (CV + R)
#define VALUES_PER_CELL 3

__kernel void compute_f_from_p(__global const float *p,
                               __global const float *u,
                               __global float *fp,
                               __global float *fm,
                               const int n) {
    // Get the global cell ID of the current work item
    // Here we assume that each cell has one OpenCL thread working on it.
    int cell_id = get_global_id(0);
    // Make sure we don't go out of bounds
    if (cell_id < n) {
        float F[VALUES_PER_CELL];  
        // Compute mach number
        float rho = p[cell_id*VALUES_PER_CELL];
        float xvel = p[cell_id*VALUES_PER_CELL + 1];
        float temp = p[cell_id*VALUES_PER_CELL + 2];
        float a = sqrt(GAMMA*R*temp);
        float Mach = xvel/a;
        float P = rho*R*temp;
        // Some SHLL constants
        float Z1 = 0.5*(Mach + 1.0);
        float Z2 = 0.5*a*(1.0-Mach*Mach);
        float Z3 = 0.5*(Mach - 1.0);

        // Compute fluxes (Can fine tune this later)
        F[0] = u[cell_id*VALUES_PER_CELL + 1];                   // Mass flux
        F[1] = xvel*u[cell_id*VALUES_PER_CELL + 1] + P;
        F[2] = xvel*u[cell_id*VALUES_PER_CELL + 2]; // Energy flux 

        // Compute Fp (forward flux)
        // FP[:,:,0] = F[:,:,0]*Z1 + U[:,:,0]*Z2
        fp[cell_id*VALUES_PER_CELL + 0] = F[0]*Z1 + u[cell_id*VALUES_PER_CELL + 0]*Z2;
        fp[(cell_id*VALUES_PER_CELL)+1] = F[1]*Z1 + u[cell_id*VALUES_PER_CELL + 1]*Z2;
        fp[(cell_id*VALUES_PER_CELL)+2] = F[2]*Z1 + u[cell_id*VALUES_PER_CELL + 2]*Z2;

        // Compute Fm (backward flux)
        // FM[:,:,0] = -F[:,:,0]*Z3 - U[:,:,0]*Z2
        fm[cell_id*VALUES_PER_CELL + 0] = -F[0]*Z3 - u[cell_id*VALUES_PER_CELL + 0]*Z2;
        fm[(cell_id*VALUES_PER_CELL)+1] = -F[1]*Z3 - u[cell_id*VALUES_PER_CELL + 1]*Z2;
        fm[(cell_id*VALUES_PER_CELL)+2] = -F[2]*Z3 - u[cell_id*VALUES_PER_CELL + 2]*Z2;
    }
}