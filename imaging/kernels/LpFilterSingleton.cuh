#ifndef CPP_EXAMPLE_KERNELS_LPFILTERSINGLETON_CUH
#define CPP_EXAMPLE_KERNELS_LPFILTERSINGLETON_CUH

#include "Kernel.cuh"

namespace imaging {

#define MAX_CIC_SIZE 512

__device__ __constant__ float gpuCicCoefficients[MAX_CIC_SIZE];

__global__ void gpuFirLp(
        float2 *__restrict__ output, const float2 *__restrict__ input,
        const int nSamples, const int totalNSamples, const int kernelWidth) {

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int ch = idx / nSamples;
    int sample = idx % nSamples;

    extern __shared__ char sharedMemory[];

    float2 *cachedInputData = (float2 *) sharedMemory;
    // Cached input data stores all the input data which is convolved with given
    // filter.
    // That means, there should be enough input data from the last thread in
    // the thread group to compute convolution.
    // Thus the below condition localIdx < (blockDim.x + kernelWidth)
    // Cache input.
    for (int i = sample - kernelWidth / 2 - 1, localIdx = threadIdx.x;
         localIdx <
         (kernelWidth + blockDim.x); i += blockDim.x, localIdx += blockDim.x) {
        if (i < 0 || i >= nSamples) {
            cachedInputData[localIdx] = make_float2(0.0f, 0.0f);
        } else {
            cachedInputData[localIdx] = input[ch * nSamples + i];
        }
    }
    __syncthreads();
    if (idx >= totalNSamples) {
        return;
    }
    float2 result = make_float2(0.0f, 0.0f);

    int localN = threadIdx.x + kernelWidth;
    for (int i = 0; i < kernelWidth; ++i) {
        result.x += cachedInputData[localN - i].x * gpuCicCoefficients[i];
        result.y += cachedInputData[localN - i].y * gpuCicCoefficients[i];
    }
    output[idx] = result;
}

/**
 * FIR filter. NOTE: there should be only one instance of this kernel in a single
 * imaging pipeline.
 * The constraint on the number of instances is due to the usage of
 * global constant memory to store filter coefficients.
 */
class LpFilterSingleton : public Kernel {
public:

    explicit LpFilterSingleton(const std::vector<float> coefficients) {
        this->nCoefficients = coefficients.size();
        CUDA_ASSERT(cudaMemcpyToSymbol(
                gpuCicCoefficients,
                coefficients.data(),
                coefficients.size() * sizeof(float),
                0, cudaMemcpyHostToDevice));
    }

    KernelInitResult prepare(const KernelInitContext &ctx) override {
        auto &inputShape = ctx.getInputShape();
        auto inputDtype = ctx.getDataType();

        if (inputShape.size() != 3) {
            throw std::runtime_error(
                    "Currently fir filter works only with 3D arrays");
        }
        this->totalNSamples = inputShape[0] * inputShape[1] * inputShape[2];
        this->nSamples = inputShape[2];
        return KernelInitResult(
                inputShape, inputDtype, ctx.getInputSamplingFrequency());
    }

    void process(NdArray *output, const NdArray *input,
                 cudaStream_t &stream) override {
        dim3 filterBlockDim(512);
        dim3 filterGridDim(
                (this->totalNSamples + filterBlockDim.x - 1) /
                filterBlockDim.x);
        unsigned sharedMemSize =
                (filterBlockDim.x + nCoefficients) * sizeof(float2);
        gpuFirLp<<<filterGridDim, filterBlockDim, sharedMemSize, stream >>>(
                output->getPtr<float2>(), input->getConstPtr<float2>(),
                this->nSamples, this->totalNSamples, this->nCoefficients);
        CUDA_ASSERT(cudaGetLastError());
    }

private:
    unsigned totalNSamples{0};
    unsigned nSamples{0};
    unsigned nCoefficients{0};
};

}

#endif //CPP_EXAMPLE_KERNELS_LPFILTERSINGLETON_CUH
