#ifndef CPP_EXAMPLE_KERNELS_RECONSTRUCT_LRI_PWI_CUH
#define CPP_EXAMPLE_KERNELS_RECONSTRUCT_LRI_PWI_CUH

#include "../NdArray.h"
#include <cuda_runtime.h>


namespace imaging {

__constant__ float xElemConst[1024];

#define M_PI 3.14159265f

#define MAX_TXS 256

// Samples offset: specifies the first sample from which the image will be reconstructed.
//
__device__ __constant__ float gpuInitDelays[MAX_TXS];

__forceinline__ __device__ float ownHypotf(float x, float y) {
    return sqrtf(x * x + y * y);
}

__device__ __forceinline__ float2 interpolate1d(const float2 *input, const float iSamp, const int offset = 0) {
    int idx = floorf(iSamp);
    float ratio = iSamp - idx;
    float2 a = input[offset + idx];
    float2 b = input[offset + idx + 1];
    float2 result;
    result.x = (1.0f - ratio) * a.x + ratio * b.x;
    result.y = (1.0f - ratio) * a.y + ratio * a.y;
    return result;
}

__global__ void iqRaw2LriPwi(
        float2 *output, const float2 *input,
        float const *zPix, float const *xPix,
        int const nZPix, int const nXPix,
        float const *txAng,
        float const minRxTang, float const maxRxTang,
        float const fs, float const fn,
        float const sos,
        unsigned const nSamp, unsigned const nElem, unsigned const nTx) {
    int z = blockIdx.x * blockDim.x + threadIdx.x;
    int x = blockIdx.y * blockDim.y + threadIdx.y;

    if (z >= nZPix || x >= nXPix) {
        return;
    }
    float txDist, rxDist, rxTang, txApod, rxApod, time, iSamp;
    float modSin, modCos, sampRe, sampIm, pixRe, pixIm, pixWgh;
    float const omega = 2 * M_PI * fn;
    float const sosInv = 1 / sos;
    float const zDistInv = 1 / zPix[z];

    for (int iTx = 0; iTx < nTx; ++iTx) {
        /* PWI */
        float r1 = (xPix[x] - xElemConst[0]) * cosf(txAng[iTx]) -
                   zPix[z] * sinf(txAng[iTx]);
        float r2 = (xPix[x] - xElemConst[nElem - 1]) * cosf(txAng[iTx]) -
                   zPix[z] * sinf(txAng[iTx]);

        txDist = xPix[x] * sinf(txAng[iTx]) + zPix[z] * cosf(txAng[iTx]);
        txApod = (r1 >= 0.f && r2 <= 0.f) ? 1.f : 0.f;



        float initDelay = gpuInitDelays[iTx];

        pixRe = 0.f;
        pixIm = 0.f;
        pixWgh = 0.f;
        for (int iElem = 0; iElem < nElem; ++iElem) {
            rxDist = ownHypotf(zPix[z], xPix[x] - xElemConst[iElem]);
            rxTang = (xPix[x] - xElemConst[iElem]) * zDistInv;

            if (rxTang < minRxTang || rxTang > maxRxTang) continue;
            rxApod = 1.0f;
            time = (txDist + rxDist) * sosInv + initDelay;
            iSamp = time * fs;

            if (iSamp <= 0.f || iSamp > static_cast<float>(nSamp - 1)) continue;

            float2 iqSamp = interpolate1d(input, iSamp, iTx*nElem*nSamp + iElem*nSamp);
            sampRe = iqSamp.x;
            sampIm = iqSamp.y;

            __sincosf(omega * time, &modSin, &modCos);

            pixRe += (sampRe * modCos - sampIm * modSin) * rxApod;
            pixIm += (sampRe * modSin + sampIm * modCos) * rxApod;
            pixWgh += rxApod;

        }
        if(pixWgh != 0) {
            output[z + x * nZPix + iTx * nZPix * nXPix].x = pixRe / pixWgh * txApod;
            output[z + x * nZPix + iTx * nZPix * nXPix].y = pixIm / pixWgh * txApod;
        }
    }
}

class ReconstructLriPwi : public Kernel {
public:

    ReconstructLriPwi(
            const std::vector<float> &zPix, const std::vector<float> &xPix,
            const std::vector<float> &txAngle,
            float minRxTang, float maxRxTang,
            unsigned startSample,
            float txFrequency, float speedOfSound, float pitch, float txNPeriods)
            : txFrequency(txFrequency),
              speedOfSound(speedOfSound),
              minRxTang(minRxTang), maxRxTang(maxRxTang),
              startSample(startSample),
              nZPix(zPix.size()), nXPix(xPix.size()), nTx(txAngle.size()),
              pitch(pitch), txNPeriods(txNPeriods),
              txAngle(txAngle) {

        if (nTx > MAX_TXS) {
            throw std::runtime_error("Number of transmits exceed 256");
        }

        CUDA_ASSERT(cudaMalloc(&zPixGpu, zPix.size() * sizeof(float)));
        CUDA_ASSERT(
                cudaMemcpy(zPixGpu, zPix.data(), zPix.size() * sizeof(float),
                           cudaMemcpyHostToDevice));
        CUDA_ASSERT(cudaMalloc(&xPixGpu, xPix.size() * sizeof(float)));
        CUDA_ASSERT(
                cudaMemcpy(xPixGpu, xPix.data(), xPix.size() * sizeof(float),
                           cudaMemcpyHostToDevice));
        CUDA_ASSERT(cudaMalloc(&txAngGpu, nTx * sizeof(float)));
        CUDA_ASSERT(cudaMemcpy(txAngGpu, txAngle.data(), nTx * sizeof(float),
                               cudaMemcpyHostToDevice));
    }

    KernelInitResult prepare(const KernelInitContext &ctx) override {
        int device;
        auto &inputShape = ctx.getInputShape();
        auto inputDtype = ctx.getDataType();

        this->nElements = inputShape[1];
        this->nSamples = inputShape[2];
        this->samplingFrequency = ctx.getInputSamplingFrequency();

        // x pos of elements:
        std::vector<float> xElemPos(nElements, 0.0f);
        float currentPos = -((float) (nElements - 1)) / 2 * pitch;
        for (int i = 0; i < this->nElements; ++i) {
            xElemPos[i] = currentPos;
            currentPos += pitch;
        }
        CUDA_ASSERT(cudaMemcpyToSymbol(xElemConst, xElemPos.data(), xElemPos.size() * sizeof(float)));

        // Determine initial delays for each TX/RX.
        float burstFactor = txNPeriods / (2 * txFrequency);
        std::vector<float> initDelays(nTx, 0);
        for (int i = 0; i < nTx; ++i) {
            initDelays[i] =
            -(startSample/65e6f) // start sample (via nominal sampling frequency)
            + 0.5f*(nElements-1)*pitch*abs(tanf(txAngle[i]))/speedOfSound // TX delay of the aperture's center
            + burstFactor;
        }
        CUDA_ASSERT(cudaMemcpyToSymbol(
                gpuInitDelays,
                initDelays.data(), initDelays.size() * sizeof(float),
                0, cudaMemcpyHostToDevice));

        return KernelInitResult(
                {this->nTx, (unsigned) this->nXPix, (unsigned) this->nZPix},
                inputDtype,
                ctx.getInputSamplingFrequency());
    }

    void process(NdArray *output, const NdArray *input,
                 cudaStream_t &stream) override {
        const dim3 block{16, 16, 1};
        dim3 grid{(unsigned int) ceilf(static_cast<float>(nZPix) / static_cast<float>(block.x)),
                  (unsigned int) ceilf(static_cast<float>(nXPix) / static_cast<float>(block.y)),
                  1};

        iqRaw2LriPwi<<<grid, block, 0, stream >>>(
                output->getPtr<float2>(), input->getConstPtr<float2>(),
                        zPixGpu, xPixGpu,
                        nZPix, nXPix,
                        txAngGpu,
                        minRxTang, maxRxTang,
                        samplingFrequency, txFrequency, speedOfSound,
                        nSamples, nElements, nTx);
        CUDA_ASSERT(cudaGetLastError());
    }

private:
    float *zPixGpu, *xPixGpu, *txAngGpu;
    size_t nZPix, nXPix;
    float minRxTang, maxRxTang;
    unsigned nTx, nElements, nSamples, startSample;
    float samplingFrequency, txFrequency, speedOfSound, pitch, txNPeriods;
    std::vector<float> txAngle;
};

}

#endif //CPP_EXAMPLE_KERNELS_RECONSTRUCT_LRI_PWI_CUH
