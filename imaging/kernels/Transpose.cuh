#ifndef CPP_EXAMPLE_KERNELS_TRANSPOSE_H
#define CPP_EXAMPLE_KERNELS_TRANSPOSE_H

#include "Kernel.cuh"
#include "../NdArray.h"

namespace imaging {

#define GPU_TRANSPOSE_TILE_DIM 32

__global__ void
gpuTranspose(short *out, const short *in, const unsigned width,
             const unsigned height) {

    __shared__ short tile[GPU_TRANSPOSE_TILE_DIM][GPU_TRANSPOSE_TILE_DIM + 1];

    unsigned xIndex = blockIdx.x * GPU_TRANSPOSE_TILE_DIM + threadIdx.x;
    unsigned yIndex = blockIdx.y * GPU_TRANSPOSE_TILE_DIM + threadIdx.y;
    unsigned zIndex = blockIdx.z;
    unsigned index_in = xIndex + yIndex * width + zIndex * width * height;

    if ((xIndex < width) && (yIndex < height)) {
        tile[threadIdx.y][threadIdx.x] = in[index_in];
    }

    __syncthreads();

    xIndex = blockIdx.y * GPU_TRANSPOSE_TILE_DIM + threadIdx.x;
    yIndex = blockIdx.x * GPU_TRANSPOSE_TILE_DIM + threadIdx.y;
    unsigned index_out = xIndex + yIndex * height + zIndex * width * height;

    if ((xIndex < height) && (yIndex < width)) {
        out[index_out] = tile[threadIdx.x][threadIdx.y];
    }
}

/**
 * A simple transposition of the two last axes.
 * Currently works only with 3D arrays.
 */
class Transpose : public Kernel {
public:
    Transpose() = default;

    KernelInitResult prepare(const KernelInitContext &ctx) override {
        auto &inputShape = ctx.getInputShape();
        auto inputDtype = ctx.getDataType();

        if (inputShape.size() != 3) {
            throw std::runtime_error(
                    "Currently transpose works only with 3D arrays");
        }
        this->nFrames = inputShape[0];
        this->nSamples = inputShape[1];
        this->nChannels = inputShape[2];

        return KernelInitResult(
                {nFrames, nChannels, nSamples}, inputDtype,
                ctx.getInputSamplingFrequency());
    }

    void process(NdArray *output, const NdArray *input,
                 cudaStream_t &stream) override {
        dim3 block(GPU_TRANSPOSE_TILE_DIM, GPU_TRANSPOSE_TILE_DIM);
        dim3 grid((nChannels - 1) / block.x + 1,
                  (nSamples - 1) / block.y + 1,
                  nFrames);
        gpuTranspose<<<grid, block, 0, stream>>>(
                output->getPtr<short>(), input->getConstPtr<short>(),
                nChannels, nSamples);
        CUDA_ASSERT(cudaGetLastError());
    }

private:
    // Output shape
    unsigned nFrames{}, nSamples{}, nChannels{};
};
}

#endif //CPP_EXAMPLE_KERNELS_TRANSPOSE_H