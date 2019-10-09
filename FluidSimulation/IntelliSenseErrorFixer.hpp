// IntelliSense error fixer
// This header should be included before AND after all your other includes

// This will allow IntelliSense see what it wants
#if !defined(__CUDACC__)

#define __CUDACC__
#define ONINTELLISENSE

#endif /* not __CUDACC__ */

#if defined(ONINTELLISENSE)

// For some reason this redefine is required
// Obviously nonesense but the CUDA compiler will not see it
#define __host__ extern
#define __device__ extern
#define __global__ extern
#define __shared__ extern
#define __constant__ extern
#define __managed__ extern

#endif /*  ONINTELLISENSE */