#ifndef CPP_EXAMPLE_PIPELINE_H
#define CPP_EXAMPLE_PIPELINE_H

#include <utility>
#include <vector>
#include <chrono>
#include <algorithm>
#include <functional>
#include "DataType.h"

// Win32/64 declspec exports
#if defined(_WIN32)
#if defined(IMAGING_BUILD_STAGE)
#define IMAGING_CPP_EXPORT __declspec(dllexport)
#else
#define IMAGING_CPP_EXPORT __declspec(dllimport)
#endif
#else
#define IMAGING_CPP_EXPORT
#endif

namespace imaging {

class Pipeline {
public:
    virtual ~Pipeline() = default;

    virtual void process(int16_t *data,
                         void (*processingCallback)(void *),
                         void (*hostRfReleaseFunction)(void *),
                         void *releasedRfElement) = 0;

    virtual const std::vector<unsigned> &getOutputShape() const = 0;
    virtual DataType getOutputDataType() const = 0;
};

IMAGING_CPP_EXPORT
std::shared_ptr<Pipeline> createPwiImagingPipeline(
        const std::vector<unsigned> &inputShape,
        int8_t *fcmChannels, uint16_t *fcmFrames,
        unsigned nTx, unsigned nElements, unsigned nSamples, unsigned startSample,
        const std::vector<float> &angles,
        float pitch, float samplingFrequency, float txFrequency, float nTxPeriods,
        float speedOfSound);
}


#endif //CPP_EXAMPLE_PIPELINE_H
