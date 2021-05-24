#ifndef CPP_EXAMPLE_METADATA_H
#define CPP_EXAMPLE_METADATA_H

struct Metadata {

    Metadata(unsigned int nSamples, float transmitFrequency,
             float samplingFrequency, float speedOfSound, float curvatureRadius,
             float pitch, float nPeriods)
        : nSamples(nSamples),
          transmitFrequency(transmitFrequency),
          samplingFrequency(samplingFrequency),
          speedOfSound(speedOfSound),
          curvatureRadius(curvatureRadius),
          pitch(pitch), nPeriods(nPeriods) {}

    unsigned nSamples;
    float transmitFrequency;
    float samplingFrequency;
    float speedOfSound;
    float curvatureRadius;
    float pitch;
    float nPeriods;
};


#endif //CPP_EXAMPLE_METADATA_H
