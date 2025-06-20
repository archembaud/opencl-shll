# OpenCL Vector Addition Demo

This is a simple OpenCL program that demonstrates parallel vector addition. The program adds two vectors `a` and `b` to produce vector `c`, where `c[i] = a[i] + b[i]` for all elements. For this demo, I'll be using the nvcc compiler.

## Files

- `vector_add.c` - Main C program that initializes OpenCL and manages the computation
- `vector_add.cl` - OpenCL kernel that performs the actual vector addition
- `Makefile` - Build configuration for easy compilation
- `README.md` - This file

## Compilation

To compile the program, simply run:
```bash
make
```

This will create an executable called `vector_add`.

## Running the Program

To run the program:
```bash
./vector_add
```

Or use the make target:
```bash
make run
```

## What the Program Does

1. **Initialization**: Sets up OpenCL platform, device, context, and command queue
2. **Data Setup**: Creates three vectors of 100 floating-point numbers each:
   - Vector `a`: contains values [0, 1, 2, ..., 99]
   - Vector `b`: contains values [0, 2, 4, ..., 198] (each element is 2× the index)
   - Vector `c`: initialized to zeros (will store the result)
3. **Kernel Execution**: Runs the OpenCL kernel in parallel to compute `c = a + b`
4. **Results**: Displays the first 10 results and the last result

## Expected Output

The program will output something like:
```
Vector Addition Results (showing first 10 elements):
c[0] = 0.00 + 0.00 = 0.00
c[1] = 1.00 + 2.00 = 3.00
c[2] = 2.00 + 4.00 = 6.00
c[3] = 3.00 + 6.00 = 9.00
c[4] = 4.00 + 8.00 = 12.00
c[5] = 5.00 + 10.00 = 15.00
c[6] = 6.00 + 12.00 = 18.00
c[7] = 7.00 + 14.00 = 21.00
c[8] = 8.00 + 16.00 = 24.00
c[9] = 9.00 + 18.00 = 27.00
...
c[99] = 99.00 + 198.00 = 297.00
Vector addition completed successfully!
```

## Cleaning Up

To remove compiled files:
```bash
make clean
```

## Troubleshooting

If you encounter compilation errors:
1. Make sure OpenCL development libraries are installed
2. Check that your system has OpenCL-compatible hardware (GPU or CPU)
3. Verify that the OpenCL runtime is properly installed

If you get runtime errors:
1. Ensure your GPU drivers are up to date
2. Check that OpenCL is supported on your hardware
3. Some systems may require additional OpenCL runtime packages 