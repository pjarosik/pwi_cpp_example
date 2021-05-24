#ifndef CPP_EXAMPLE_KERNELS_KERNELINITCONTEXT_H
#define CPP_EXAMPLE_KERNELS_KERNELINITCONTEXT_H

#include <utility>

#include "../NdArray.h"
namespace imaging {
class KernelInitContext {
public:
    KernelInitContext(NdArray::DataShape input_shape,
                      NdArray::DataType data_type,
                      float sampling_frequency)
            : inputShape(std::move(input_shape)),
              dataType(data_type),
              samplingFrequency(sampling_frequency) {}

    float getInputSamplingFrequency() const {
        return samplingFrequency;
    }

    const NdArray::DataShape &getInputShape() const {
        return inputShape;
    }

    NdArray::DataType getDataType() const {
        return dataType;
    }

    float getSamplingFrequency() const {
        return samplingFrequency;
    }

private:
    NdArray::DataShape inputShape;
    NdArray::DataType dataType;
    float samplingFrequency;
};
}
#endif //CPP_EXAMPLE_KERNELS_KERNELINITCONTEXT_H
