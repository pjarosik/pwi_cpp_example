#ifndef CPP_EXAMPLE_KERNELS_ENVELOPEDETECTION_CUH
#define CPP_EXAMPLE_KERNELS_ENVELOPEDETECTION_CUH

#include "../NdArray.h"
#include "KernelInitResult.h"
#include "KernelInitContext.h"
#include "Kernel.cuh"

namespace imaging {
__global__ void gpuEnvelopeDetection(float *output, const float2 *input,
                                     const unsigned totalNSamples) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= totalNSamples) {
        return;
    }
    float2 value = input[idx];
    output[idx] = hypotf(value.x, value.y);
}

class EnvelopeDetection : public Kernel {
public:
    EnvelopeDetection() = default;

    KernelInitResult prepare(const KernelInitContext &ctx) override {
        auto &inputShape = ctx.getInputShape();

        if (inputShape.size() != 2) {
            throw std::runtime_error(
                    "Currently envelope detection works only with 2D arrays");
        }
        this->totalNSamples = inputShape[0] * inputShape[1];
        return KernelInitResult({inputShape[0], inputShape[1]},
                                NdArray::DataType::FLOAT32,
                                ctx.getInputSamplingFrequency());
    }

    void process(NdArray *output, const NdArray *input,
                 cudaStream_t &stream) override {
        dim3 filterBlockDim(512);
        dim3 filterGridDim(
                (this->totalNSamples + filterBlockDim.x - 1) /
                filterBlockDim.x);
        gpuEnvelopeDetection<<<filterGridDim, filterBlockDim, 0, stream >>>(
                output->getPtr<float>(), input->getConstPtr<float2>(),
                this->totalNSamples);
        CUDA_ASSERT(cudaGetLastError());
    }

private:
    unsigned totalNSamples{0};
};

}

#endif //CPP_EXAMPLE_KERNELS_QUADRATUREDEMODULATION_CUH
