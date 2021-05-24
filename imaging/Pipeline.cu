#include <cmath>
#include <iostream>
#include <fstream>


#include "kernels/RemapToLogicalOrder.cuh"
#include "kernels/Transpose.cuh"
#include "kernels/FirFilterSingleton.cuh"
#include "kernels/QuadratureDemodulation.cuh"
#include "kernels/Decimation.cuh"
#include "kernels/LpFilterSingleton.cuh"
#include "kernels/EnvelopeDetection.cuh"
#include "kernels/ToBmode.cuh"
#include "kernels/ReconstructLriPwi.cuh"
#include "kernels/Sum.h"
#include "Pipeline.cuh"
#include "Metadata.h"
#include "NdArray.h"

namespace imaging {
#include "../common.h"

// grid OX coordinates
constexpr float X_PIX_L = -19.0e-3f, X_PIX_R = 19.0e-3f, X_PIX_STEP = 0.1e-3;
// grid OZ coordinates
constexpr float Z_PIX_L = 5e-3f, Z_PIX_R = 42.5e-3f, Z_PIX_STEP = 0.1e-3;

// Hanning window, band (0.5, 1.5)*TX_FREQUENCY, order 64
const std::vector<float> BANDPASS_FILTER_COEFFS
    {-2.73297719e-05, 1.71755184e-03, 1.02162832e-04, -1.08044042e-04,
     2.15183914e-04, -3.22957878e-03, -5.14532875e-04, 4.62438158e-04,
     -8.02126191e-04, 6.91522057e-03, 1.79774703e-03, -1.42160546e-03,
     2.20882764e-03, -1.33118718e-02, -4.89203646e-03, 3.47650911e-03,
     -4.96440929e-03, 2.32521134e-02, 1.14209381e-02, -7.41976146e-03,
     9.95996840e-03, -3.88851306e-02, -2.50219179e-02, 1.51484961e-02,
     -1.96856739e-02, 6.84264416e-02, 5.88343927e-02, -3.49935763e-02,
     4.81262179e-02, -1.75943846e-01, -2.64008947e-01, 3.43890937e-01,
     3.43890937e-01, -2.64008947e-01, -1.75943846e-01, 4.81262179e-02,
     -3.49935763e-02, 5.88343927e-02, 6.84264416e-02, -1.96856739e-02,
     1.51484961e-02, -2.50219179e-02, -3.88851306e-02, 9.95996840e-03,
     -7.41976146e-03, 1.14209381e-02, 2.32521134e-02, -4.96440929e-03,
     3.47650911e-03, -4.89203646e-03, -1.33118718e-02, 2.20882764e-03,
     -1.42160546e-03, 1.79774703e-03, 6.91522057e-03, -8.02126191e-04,
     4.62438158e-04, -5.14532875e-04, -3.22957878e-03, 2.15183914e-04,
     -1.08044042e-04, 1.02162832e-04, 1.71755184e-03, -2.73297719e-05};

// Low-pass CIC filter cooefficients.
const std::vector<float> LOWPASS_FILTER_COEFFS{1, 2, 3, 4, 3, 2, 1};

class PipelineImpl : public Pipeline {
public:
    PipelineImpl(std::vector<Kernel::Handle> kernels,
                 NdArray::DataShape inputShape,
                 NdArray::DataType inputDtype, float samplingFrequency)
            : kernels(std::move(kernels)),
              inputShape(std::move(inputShape)), inputDtype(inputDtype),
              samplingFrequency(samplingFrequency) {
        inputGpu = NdArray{this->inputShape, this->inputDtype, true};
        CUDA_ASSERT(cudaStreamCreate(&processingStream));
        CUDA_ASSERT(cudaStreamCreate(&inputDataStream));
        prepare();
    }

    ~PipelineImpl() override {
        CUDA_ASSERT_NO_THROW(cudaStreamDestroy(processingStream));
        CUDA_ASSERT_NO_THROW(cudaStreamDestroy(inputDataStream));
    }

    void prepare() {
        float currentSamplingFrequency = samplingFrequency;
        NdArray::DataShape shape = inputShape;
        NdArray::DataType dataType = inputDtype;
        for (auto &kernel: kernels) {
            KernelInitContext kernelInitContext(
                    shape, dataType, currentSamplingFrequency);
            auto prepareOutput = kernel->prepare(kernelInitContext);
            kernelOutputs.emplace_back(prepareOutput.GetOutputShape(),
                                       prepareOutput.GetOutputDtype(),
                                       true);
            shape = prepareOutput.GetOutputShape();
            dataType = prepareOutput.GetOutputDtype();
            currentSamplingFrequency = prepareOutput.getSamplingFrequency();
        }
        outputShape = shape;
        outputDtype = dataType;
        outputHost = NdArray(shape, dataType, false);
    }

    void process(int16_t *data,
                 void (*processingCallback)(void *),
                 void (*hostRfReleaseFunction)(void *),
                 void *releasedElement) override {
        // NOTE: data transfers H2D, D2H and processing are intentionally
        // serialized here into a single 'processingStream', for the sake
        // of simplicity.
        // Normally, n-element buffers should probably used (with some
        // additional synchronization or overwrite detection) as a common
        // memory area for communication between RF data producer and
        // consumer.

        // Wrap pointer to the input data into NdArray object.
        NdArray inputHost{data, inputShape, inputDtype, false};
        // Transfer data H2D.
        CUDA_ASSERT(cudaMemcpyAsync(
                inputGpu.getPtr<void>(),
                inputHost.getPtr<void>(), inputHost.getNBytes(),
                cudaMemcpyHostToDevice, processingStream));
        // Release host RF buffer element after transferring the data.
        CUDA_ASSERT(cudaLaunchHostFunc(processingStream, hostRfReleaseFunction,
                                       releasedElement));

        // Execute a sequence of pipeline kernels.
        NdArray *currentInput = &inputGpu;
        NdArray *currentOutput = &inputGpu;
        for (size_t kernelNr = 0; kernelNr < kernels.size(); ++kernelNr) {
            currentOutput = &(kernelOutputs[kernelNr]);
            kernels[kernelNr]->process(currentOutput, currentInput,
                                       processingStream);
            currentInput = currentOutput;
        }
        // Transfer data D2H.
        CUDA_ASSERT( cudaMemcpyAsync(
                outputHost.getPtr<void>(),
                currentOutput->getPtr<void>(), outputHost.getNBytes(),
            cudaMemcpyDeviceToHost, processingStream));
        CUDA_ASSERT(cudaStreamSynchronize(processingStream));
        processingCallback(outputHost.getPtr<void>());
        // There seems to be some issues when calling opencv::imshow in cuda callback,
        // so I had to use cudaStreamSynchronize here.
//        CUDA_ASSERT(cudaLaunchHostFunc(processingStream, processingCallback, outputHost.getPtr<void>()));
    }

    const NdArray::DataShape &getOutputShape() const {
        return outputShape;
    }

    DataType getOutputDataType() const {
        return outputDtype;
    }

private:
    std::vector<Kernel::Handle> kernels;
    NdArray inputGpu, outputHost;
    std::vector<NdArray> kernelOutputs;
    NdArray::DataShape inputShape;
    NdArray::DataType inputDtype;
    NdArray::DataShape outputShape;
    NdArray::DataType outputDtype;
    float samplingFrequency;
    cudaStream_t processingStream{}, inputDataStream;

    void (*gpuElementReleaseFunction)(void *);
};

#define ADD_KERNEL(kernelName, ...) kernels.push_back(std::make_unique<kernelName>(__VA_ARGS__))

std::shared_ptr<Pipeline> createPwiImagingPipeline(
        const std::vector<unsigned> &inputShape,
        int8_t *fcmChannels, uint16_t *fcmFrames,
        unsigned nTx, unsigned nElements, unsigned nSamples, unsigned startSample,
        const std::vector<float> &angles,
        float pitch, float samplingFrequency, float txFrequency,
        float txNPeriods,
        float speedOfSound) {

    // Wrap frame channel mapping into NdArrays.
    std::vector<unsigned> fcmArrayShape{static_cast<unsigned int>(nTx), nElements};
    NdArray fcmChannelsArray{fcmChannels, fcmArrayShape, NdArray::DataType::INT8, false};
    NdArray fcmFramesArray{fcmFrames, fcmArrayShape, NdArray::DataType::UINT16, false};

    // Output image grid points: (xPix[0], zPix[0]), (xPix[1], zPix[1]), ...
    // xPix: OX grid point coordinates
    // zPix: OZ grid point coordinates
    std::vector<float> xPix = arange(X_PIX_L, X_PIX_R, X_PIX_STEP);
    std::vector<float> zPix = arange(Z_PIX_L, Z_PIX_R, Z_PIX_STEP);

    std::vector<Kernel::Handle> kernels;

    // Reorder the input data from (nTxMuxed*nSamples*nUs4OEM, 32)
    // to format (nTx, nSamples, nElements)
    //
    // input dimensions: (nTxMuxed*nSamples*nUs4OEM, 32)
    // output dimensions: (nTx, nSamples, nElements)
    // output data type: int16
    ADD_KERNEL(RemapToLogicalOrder,
               std::move(fcmChannelsArray.copyToDevice()),
               std::move(fcmFramesArray.copyToDevice()),
               nSamples);
    // Transpose data from (nTx, nSamples, nElements) to (nTx, nElements, nSamples)
    //
    // input dimensions: (nTx, nSamples, nElements)
    // output dimensions: (nTx, nElements, nSamples)
    // output data type: int16
    ADD_KERNEL(Transpose);
    // Fir filter bandpass: [0.5 * tx frequency, 1.5 * tx frequency]
    //
    // input dimensions: (nTx, nElements, nSamples)
    // output dimensions: (nTx, nElements, nSamples)
    // output data type: float32
    ADD_KERNEL(FirFilterSingleton, BANDPASS_FILTER_COEFFS);

    // --- DDC
    // Demodulate data to I/Q
    //
    // input dimensions: (nTx, nElements, nSamples)
    // output dimensions: (nTx, nSamples, nSamples)
    // output data type: (float32, float32)
    ADD_KERNEL(QuadratureDemodulation, txFrequency);
    // Low-pass FIR filter.
    //
    // input dimensions: (nTx, nElements, nSamples)
    // output dimensions: (nTx, nSamples, nSamples)
    // output data type: (float32, float32)
    ADD_KERNEL(LpFilterSingleton, LOWPASS_FILTER_COEFFS);
    // Downsample the data by a factor of 4
    //
    // input dimensions: (nTx, nElements, nSamples)
    // output dimensions: (nTx, nSamples, nSamples/4)
    // output data type: (float32, float32)
    ADD_KERNEL(Decimation, 4);

    // Rx beamforming for Plane Wave Imaging
    // Reconstructs LRIs from the input RF data.
    //
    // input dimensions: (nTx, nSamples, nSamples/4)
    // output dimensions: Lris: (nTx, xGridSize, zGridSize)
    // output data type: (float32, float32)
    // maxTang value determines RX apodization max angle.
    float maxTang = tanf(asinf(std::min(1.0f, (float) (speedOfSound / txFrequency * 2 / 3) / pitch)));
    ADD_KERNEL(ReconstructLriPwi, zPix, xPix, angles,
               -maxTang, maxTang,
               startSample,
               txFrequency, speedOfSound, pitch, txNPeriods);
    // Sum input data along the first dimension.
    // This is a very simple conversion from an array of LRIs to HRI
    //
    // input dimensions: LRIs: (nTx, xGridSize, zGridSize)
    // output dimensions: HRI: (xGridSize, zGridSize)
    // output data type: (float32, float32)
    ADD_KERNEL(Sum);
    // Compute absolute value for the input complex data.
    //
    // input dimensions: HRI: (xGridSize, zGridSize)
    // output dimensions: (xGridSize, zGridSize)
    // output data type: float32
    ADD_KERNEL(EnvelopeDetection);
    // ToBmode: log compression, dynamic range adjustment and map to uint8.
    //
    // input dimensions:  (xGridSize, zGridSize)
    // output dimensions: (xGridSize, zGridSize)
    // output data type: uint8
    ADD_KERNEL(ToBmode, 20, 80);


    std::shared_ptr<Pipeline> result = std::make_shared<PipelineImpl>(
            std::move(kernels),
            inputShape,
            NdArray::DataType::INT16,
            samplingFrequency);
    return result;
}
}



