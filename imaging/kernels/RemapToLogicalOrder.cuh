#ifndef CPP_EXAMPLE_KERNELS_REMAPTOLOGICALORDER_CUH
#define CPP_EXAMPLE_KERNELS_REMAPTOLOGICALORDER_CUH

#include "../CudaUtils.cuh"
#include "../NdArray.h"
#include "KernelInitResult.h"
#include "KernelInitContext.h"
#include "Kernel.cuh"

namespace imaging {
__global__ void arrusRemap(short *out, const short *in,
                           const short *fcmFrames,
                           const char *fcmChannels,
                           const unsigned nFrames,
                           const unsigned nSamples,
                           const unsigned nChannels) {
    int x = blockIdx.x * 32 + threadIdx.x; // logical channel
    int y = blockIdx.y * 32 + threadIdx.y; // logical sample
    int z = blockIdx.z; // logical frame
    if (x >= nChannels || y >= nSamples || z >= nFrames) {
        // outside the range
        return;
    }
    int indexOut = x + y * nChannels + z * nChannels * nSamples;
    int physicalChannel = fcmChannels[x + nChannels * z];
    if (physicalChannel < 0) {
        // channel is turned off
        return;
    }
    int physicalFrame = fcmFrames[x + nChannels * z];
    // 32 - number of channels in the physical mapping
    int indexIn = physicalChannel + y * 32 + physicalFrame * 32 * nSamples;
    out[indexOut] = in[indexIn];
}

class RemapToLogicalOrder : public Kernel {
public:
    RemapToLogicalOrder(NdArray fcmChannels, NdArray fcmFrames, unsigned nSamples)
            : fcmChannels(std::move(fcmChannels)),
              fcmFrames(std::move(fcmFrames)),
              nSamples(nSamples) {
        nFrames = this->fcmFrames.getShape()[0];
        nChannels = this->fcmFrames.getShape()[1];
        fcmFramesPtr = this->fcmFrames.getPtr<short>();
        fcmChannelsPtr = this->fcmChannels.getPtr<char>();
    }

    KernelInitResult prepare(const KernelInitContext &ctx) override {
        return KernelInitResult({nFrames, nSamples, nChannels},
                                NdArray::DataType::INT16,
                                ctx.getInputSamplingFrequency());
    }

    void process(NdArray *output, const NdArray *input,
                 cudaStream_t &stream) override {
        dim3 block(BLOCK_TILE_DIM, BLOCK_TILE_DIM);
        dim3 grid((nChannels - 1) / block.x + 1,
                  (nSamples - 1) / block.y + 1,
                  nFrames);
        arrusRemap<<<grid, block, 0, stream>>>(
                output->getPtr<short>(), input->getConstPtr<short>(),
                fcmFramesPtr, fcmChannelsPtr,
                nFrames, nSamples, nChannels);
        CUDA_ASSERT(cudaGetLastError());
    }

private:
    static constexpr int BLOCK_TILE_DIM = 32;

    // Output shape
    unsigned nFrames, nSamples, nChannels;
    NdArray fcmChannels, fcmFrames;
    short *fcmFramesPtr;
    char *fcmChannelsPtr;
};
}


#endif //CPP_EXAMPLE_KERNELS_REMAPTOLOGICALORDER_CUH
