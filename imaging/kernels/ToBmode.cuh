#ifndef CPP_EXAMPLE_KERNELS_TOBMODE_CUH
#define CPP_EXAMPLE_KERNELS_TOBMODE_CUH

namespace imaging {

__global__ void gpuBMode(uint8_t *output, const float *input,
                         const float minDBLimit, const float maxDBLimit,
                         const int maxThreads) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= maxThreads) {
        return;
    }
    float pix = input[idx] + 1e-9;
    pix = 20.0f * log10f(pix);
    if (isnan(pix)) {
        pix = 0.0f;
    }
    // Cut on limits
    pix = fmaxf(minDBLimit, fminf(maxDBLimit, pix));
    // TODO Do a better remapping here.
    pix = pix-minDBLimit;
    pix = pix/(maxDBLimit-minDBLimit)*255;
    output[idx] = (uint8_t)pix;
}

/**
 * Converts to decibel scale and clips to given dynamic range values.
 */
class ToBmode : public Kernel {

public:
    ToBmode(unsigned int minDbLimit, unsigned int maxDbLimit)
            : minDBLimit(minDbLimit), maxDBLimit(maxDbLimit) {}

private:

    KernelInitResult prepare(const KernelInitContext &ctx) override {
        auto &inputShape = ctx.getInputShape();
        auto inputDtype = ctx.getDataType();

        if (inputShape.size() != 2) {
            throw std::runtime_error(
                    "Currently converting to decibel scale works only with 2D arrays");
        }
        this->totalPixels = inputShape[0] * inputShape[1];
        return KernelInitResult({inputShape[0], inputShape[1]},
                                DataType::UINT8,
                                ctx.getInputSamplingFrequency());

    }

    void process(NdArray *output, const NdArray *input,
                 cudaStream_t &stream) override {
        dim3 blockDim(512);
        dim3 gridDim((this->totalPixels + blockDim.x - 1) / blockDim.x);
        gpuBMode<<<gridDim, blockDim, 0, stream >>>(
                output->getPtr<uint8_t>(), input->getConstPtr<float>(),
                this->minDBLimit, this->maxDBLimit,
                this->totalPixels);
        CUDA_ASSERT(cudaGetLastError());
    }

private:
    unsigned minDBLimit, maxDBLimit, totalPixels;

};

}

#endif //CPP_EXAMPLE_KERNELS_TOBMODE_CUH
