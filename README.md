## ISSUE: Low GPU Utilization 
When a3fe enter the stage of ensemble equilibration, the GPU load drops sharply: https://github.com/michellab/a3fe/issues/50
## Solution
- Ensure that GROMACS is configured to support CUDA acceleration instead of OPENCL.
   Currently, the version with OPENCL acceleration does not support many interactions and can lead to errors.
   Here is my Gromacs information:
```
   (base) gkxiao@master:~$ gmx --version
                         :-) GROMACS - gmx, 2025.2 (-:

Executable:   /public/gkxiao/software/gromacs/gromacs_2025.2build20250617/bin/gmx
Data prefix:  /public/gkxiao/software/gromacs/gromacs_2025.2build20250617
Working dir:  /home/gkxiao
Command line:
  gmx --version

GROMACS version:     2025.2
Precision:           mixed
Memory model:        64 bit
MPI library:         thread_mpi
OpenMP support:      enabled (GMX_OPENMP_MAX_THREADS = 128)
GPU support:         CUDA
NBNxM GPU setup:     super-cluster 2x2x2 / cluster 8 (cluster-pair splitting on)
SIMD instructions:   AVX2_256
CPU FFT library:     fftw-3.3.10-sse2-avx
GPU FFT library:     cuFFT
Multi-GPU FFT:       none
RDTSCP usage:        enabled
TNG support:         enabled
Hwloc support:       disabled
Tracing support:     disabled
C compiler:          /usr/bin/cc GNU 13.3.0
C compiler flags:    -fexcess-precision=fast -funroll-all-loops -mavx2 -mfma -Wno-missing-field-initializers -O3 -DNDEBUG
C++ compiler:        /usr/bin/c++ GNU 13.3.0
C++ compiler flags:  -fexcess-precision=fast -funroll-all-loops -mavx2 -mfma -Wno-missing-field-initializers -Wno-cast-function-type-strict SHELL:-fopenmp -O3 -DNDEBUG
BLAS library:        External - detected on the system
LAPACK library:      External - detected on the system
CUDA compiler:       /usr/local/cuda-12.6/bin/nvcc nvcc: NVIDIA (R) Cuda compiler driver;Copyright (c) 2005-2024 NVIDIA Corporation;Built on Tue_Oct_29_23:50:19_PDT_2024;Cuda compilation tools, release 12.6, V12.6.85;Build cuda_12.6.r12.6/compiler.35059454_0
CUDA compiler flags:-arch=sm_89 -O3 -DNDEBUG
CUDA driver:         12.60
CUDA runtime:        12.60
```
This is my compilation method:
```
cmake ..   -DCMAKE_C_COMPILER=/usr/bin/cc \
        -DCMAKE_CXX_COMPILER=/usr/bin/c++ \
        -DCMAKE_INSTALL_PREFIX=/public/gkxiao/software/gromacs/gromacs_2025.2build20250617 \
        -DCUDA_NVCC_FLAGS="-arch=sm_89" \
        -DGMX_CUDA_USE_UNIFIED_MEMORY=ON \
        -DGMX_MPI=OFF \
        -DGMX_THREAD_MPI=ON \
        -DGMX_OPENMP=ON \
        -DGMX_GPU=CUDA \
        -DGMX_SIMD=AVX2_256 \
        -DGMX_FFT_LIBRARY=fftw3 \
        -DGMX_BUILD_OWN_FFTW=OFF \
        -DGMX_USE_RDTSCP=ON \
        -DGMX_USE_TNG=ON \
        -DGMX_HWLOC=OFF \
        -DGMX_DOUBLE=OFF \
        -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.6 \
        -DCMAKE_BUILD_TYPE=Release
```
- Replace a3fe/run/system_prep.py with the same-named file from the attachment.
- Alternatively,you can modify the lines following line 722 as shown below.
```
    process = _BSS.Process.Gromacs(system, protocol, work_dir=work_dir)
    #For non-bonded interactions is always safe to use
    #process.setArg("-nb", "gpu")
    #Additional options (e.g., PME, bonded, update) should only be applied outside the minimization stage
    if not isinstance(protocol, _BSS.Protocol.Minimisation):
        process.setArg("-nb", "gpu")
        process.setArg("-pme", "gpu")
        process.setArg("-bonded", "gpu")
        process.setArg("-ntomp","8")
    process.start()
    process.wait()
    import time
```
