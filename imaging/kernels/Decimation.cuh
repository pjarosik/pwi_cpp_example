#ifndef CPP_EXAMPLE_KERNELS_DECIMATION_CUH
#define CPP_EXAMPLE_KERNELS_DECIMATION_CUH

#include "../NdArray.h"
#include "KernelInitResult.h"
#include "KernelInitContext.h"
#include "Kernel.cuh"

namespace imaging {

__global__ void gpuDecimation(float2 *output, const float2 *input,
                              const unsigned nSamples,
                              const unsigned totalNSamples,
                              const unsigned decimationFactor) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx >= totalNSamples) {
        return;
    }
    int decimatedNSamples = (int) ceilf((float) nSamples / decimationFactor);
    output[idx] = input[
            (idx / decimatedNSamples) * nSamples // (transmit, channel number)
            + (idx % decimatedNSamples) * decimationFactor];
}

class Decimation : public Kernel {
public:

    KernelInitResult prepare(const KernelInitContext &ctx) override {
        auto &inputShape = ctx.getInputShape();
        auto inputDtype = ctx.getDataType();

        if (inputShape.size() != 3) {
            throw std::runtime_error(
                    "Currently decimation works only with 3D arrays");
        }
        this->nSamples = inputShape[2];
        auto outputNSamples = (unsigned) ceilf(
                (float) nSamples / decimationFactor);
        this->outputTotalNSamples =
                inputShape[0] * inputShape[1] * outputNSamples;
        return KernelInitResult(
                {inputShape[0], inputShape[1], outputNSamples},
                inputDtype,
                ctx.getInputSamplingFrequency() / decimationFactor);
    }

    void process(NdArray *output,
                 const NdArray *input,
                 cudaStream_t &stream) override {
        dim3 block(512);
        dim3 grid((outputTotalNSamples + block.x - 1) / block.x);
        gpuDecimation <<<grid, block, 0, stream >>>(
                output->getPtr<float2>(), input->getConstPtr<float2>(),
                nSamples,
                outputTotalNSamples,
                decimationFactor);
        CUDA_ASSERT(cudaGetLastError());
    }

    Decimation(unsigned decimationFactor) : decimationFactor(
            decimationFactor) {}

private:
    unsigned outputTotalNSamples{0};
    unsigned nSamples{0};
    unsigned decimationFactor;
};
}

#endif //CPP_EXAMPLE_KERNELS_DECIMATION_CUH
