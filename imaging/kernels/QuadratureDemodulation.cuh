#ifndef CPP_EXAMPLE_KERNELS_QUADRATUREDEMODULATION_CUH
#define CPP_EXAMPLE_KERNELS_QUADRATUREDEMODULATION_CUH

#include "../NdArray.h"
#include "math_constants.h"

namespace imaging {

__global__ void gpuRfToIq(float2 *output, const float *input,
                          const float sampleCoeff,
                          const unsigned nSamples,
                          const unsigned maxThreadId) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= maxThreadId) {
        return;
    }
    float rfSample = input[idx];
    int sampleNumber = idx % nSamples;
    float cosinus, sinus;
    __sincosf(sampleCoeff * sampleNumber, &sinus, &cosinus);
    float2 iq;
    iq.x = 2.0f * rfSample * cosinus;
    iq.y = 2.0f * rfSample * sinus;
    output[idx] = iq;
}

class QuadratureDemodulation : public Kernel {
public:
    QuadratureDemodulation(float transmitFrequency) :
            transmitFrequency(transmitFrequency) {}

    KernelInitResult prepare(const KernelInitContext &ctx) override {
        auto &inputShape = ctx.getInputShape();
        if (inputShape.size() != 3) {
            throw std::runtime_error(
                    "Currently demodulation works only with 3D arrays");
        }
        auto samplingFrequency = ctx.getSamplingFrequency();
        this->totalNSamples = inputShape[0] * inputShape[1] * inputShape[2];
        this->nSamples = inputShape[2];
        this->sampleCoeff =
                -2.0f * CUDART_PI_F * transmitFrequency / samplingFrequency;
        return KernelInitResult(
                inputShape, NdArray::DataType::COMPLEX64,
                ctx.getInputSamplingFrequency());
    }

    void process(NdArray *output, const NdArray *input,
                 cudaStream_t &stream) override {
        dim3 filterBlockDim(512);
        dim3 filterGridDim(
                (this->totalNSamples + filterBlockDim.x - 1) /
                filterBlockDim.x);
        gpuRfToIq<<<filterGridDim, filterBlockDim, 0, stream >>>(
                output->getPtr<float2>(), input->getConstPtr<float>(),
                this->sampleCoeff, this->nSamples, this->totalNSamples);
        CUDA_ASSERT(cudaGetLastError());
    }


private:
    unsigned totalNSamples{0};
    unsigned nSamples{0};
    unsigned nCoefficients{0};
    float samplingFrequency;
    float transmitFrequency;
    float sampleCoeff{0};
};

}

#endif //CPP_EXAMPLE_KERNELS_QUADRATUREDEMODULATION_CUH
