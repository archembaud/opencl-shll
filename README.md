# opencl-shll
A simple finite volume solver using OpenCL with the SHLL method

There are a number of folders in this repository:

* vector-demo: A simple "hello world" demonstration of OpenCL being applied to a vector computation. (Go the the [README](./vector-demo/README.md) file)
* shll-1d: The most basic 1D, first order accurate SHLL implementation possible, applied to the simulation of Sod's 1D shock tube. (Go to the [README](./shll-1d/README.md) file)


## Prerequisites

You'll need to make sure that you've got OpenCL installed. In the examples provided here, we make use of the OpenCL libraries which are included as part of the CUDA toolkit. Hence, the makefiles (described below) use the NVCC compiler. This is not required - but instructions for using OpenCL with a different compiler and OpenCL source are not shown here.

## Building the files

* Navigate to the directory you wish to build,
* Build using make:

```bash
make
```
