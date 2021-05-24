#ifndef CPP_EXAMPLE_KERNELS_KERNEL_CUH
#define CPP_EXAMPLE_KERNELS_KERNEL_CUH

#include <memory>
#include <cuda_runtime.h>
#include "KernelInitContext.h"
#include "KernelInitResult.h"
namespace imaging {
class Kernel {
public:
    typedef std::unique_ptr<Kernel> Handle;

    virtual
    KernelInitResult prepare(const KernelInitContext &ctx) = 0;

    virtual
    void
    process(NdArray *output, const NdArray *input, cudaStream_t &stream) = 0;
};
}
#endif //CPP_EXAMPLE_KERNELS_KERNEL_CUH
