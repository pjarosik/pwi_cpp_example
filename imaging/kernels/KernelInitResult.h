#ifndef CPP_EXAMPLE_KERNELS_KERNELHEADERRESULT_H
#define CPP_EXAMPLE_KERNELS_KERNELHEADERRESULT_H

#include <utility>

#include "../NdArray.h"
namespace imaging {
class KernelInitResult {
public:
    KernelInitResult(NdArray::DataShape output_shape,
                     NdArray::DataType output_dtype,
                     float sampling_frequency)
        : outputShape(std::move(output_shape)),
          outputDtype(output_dtype),
          samplingFrequency(sampling_frequency) {}

    float getSamplingFrequency() const {
        return samplingFrequency;
    }

    const NdArray::DataShape &GetOutputShape() const {
        return outputShape;
    }
    NdArray::DataType GetOutputDtype() const {
        return outputDtype;
    }
    float GetSamplingFrequency() const {
        return samplingFrequency;
    }

 private:
    NdArray::DataShape outputShape;
    NdArray::DataType outputDtype;
    float samplingFrequency;
};
}
#endif //CPP_EXAMPLE_KERNELS_KERNELHEADERRESULT_H
