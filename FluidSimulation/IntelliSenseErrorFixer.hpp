// IntelliSense error fixer
// This header should be included before AND after all your other includes

// This will allow IntelliSense see what it wants
#if !defined(__CUDACC__)

#define __CUDACC__
#define ONINTELLISENSE

#endif /* not __CUDACC__ */

#if defined(ONINTELLISENSE)

#define KERNEL_CALL(numBlocks, threadsPerBlock)

// For some reason this redefine is required
#define __host__
#define __device__
#define __global__
#define __shared__
#define __constant__
#define __managed__

#endif /*  ONINTELLISENSE */

#if !defined(ONINTELLISENSE)

#define KERNEL_CALL(numBlocks, threadsPerBlock) <<<numBlocks,threadsPerBlock>>>

#endif /*  not ONINTELLISENSE */