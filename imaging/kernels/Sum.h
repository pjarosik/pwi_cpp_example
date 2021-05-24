#ifndef CPP_EXAMPLE_KERNELS_SUM_H_
#define CPP_EXAMPLE_KERNELS_SUM_H_

#include "Kernel.cuh"

namespace imaging {

__global__ void gpuSum(float2 *__restrict__ output,
                       const float2 *__restrict__ input,
                       const unsigned ax1, const unsigned ax2,
                       const unsigned ax3) {

    int z = blockIdx.x * blockDim.x + threadIdx.x;
    int x = blockIdx.y * blockDim.y + threadIdx.y;

    if (z >= ax3 || x >= ax2) {
        return;
    }
    float2 result = make_float2(0.0f, 0.0f);
    for (int i = 0; i < ax1; ++i) {
        result.x += input[z + x * ax3 + i * ax2 * ax3].x;
        result.y += input[z + x * ax3 + i * ax2 * ax3].y;
    }
    output[z + x * ax3] = result;
}

// Sum along the first axis.
class Sum : public Kernel {
public:
    explicit Sum() = default;

    KernelInitResult prepare(const KernelInitContext &ctx) override {
        auto &inputShape = ctx.getInputShape();

        this->ax1 = inputShape[0];
        this->ax2 = inputShape[1];
        this->ax3 = inputShape[2];

        return KernelInitResult(
                {ax2, ax3}, NdArray::DataType::COMPLEX64,
                ctx.getInputSamplingFrequency());
    }

    void process(NdArray *output, const NdArray *input,
                 cudaStream_t &stream) override {
        const dim3 block{16, 16, 1};
        dim3 grid{(unsigned int) ceilf(
                static_cast<float>(ax3) / static_cast<float>(block.x)),
                  (unsigned int) ceilf(static_cast<float>(ax2) /
                                       static_cast<float>(block.y)), 1};
        gpuSum<<<grid, block, 0, stream >>>(
                output->getPtr<float2>(), input->getConstPtr<float2>(),
                this->ax1, this->ax2, this->ax3);
        CUDA_ASSERT(cudaGetLastError());
    }

private:
    unsigned ax1{}, ax2{}, ax3{};
};

}

#endif //CPP_EXAMPLE_KERNELS_SUM_H_
