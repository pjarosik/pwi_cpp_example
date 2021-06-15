#include <iostream>
#include <thread>
#include <fstream>
#include <cstdio>
#include <string>
#include <condition_variable>

#include "arrus/core/api/arrus.h"

using namespace ::arrus::session;
using namespace ::arrus::devices;
using namespace ::arrus::ops::us4r;
using namespace ::arrus::framework;

constexpr float SPEED_OF_SOUND = 1490;
constexpr unsigned DOWNSAMPLING_FACTOR = 1;
constexpr float SAMPLING_FREQUENCY = 65e6/DOWNSAMPLING_FACTOR; // [Hz]
constexpr unsigned SAMPLE_RANGE_END = 1*1024;

bool isRunning = true;

void setVoltage(Us4R *us4r) {
    try {
        unsigned voltage = 5;
        std::cout << "Please provide the voltage to set [V]" << std::endl;
        std::cin >> voltage;
        std::cout << "Got voltage: " << voltage << std::endl;
        us4r->setVoltage(voltage);
    } catch(const arrus::IllegalArgumentException& e) {
        std::cerr << e.what() << std::endl;
    }
}

std::vector<float> getLinearTGCCurve(float tgcStart, float tgcSlope,
                                     float samplingFrequency,
                                     float speedOfSound,
                                     float sampleRangeEnd) {
    std::vector<float> tgcCurve;

    float startDepth = 300.0f/samplingFrequency*speedOfSound;
    float endDepth = sampleRangeEnd/samplingFrequency*speedOfSound;
    float tgcSamplingStep = 150.0f/samplingFrequency*speedOfSound;
    float currentDepth = startDepth;

    while(currentDepth < endDepth) {
        float tgcValue = tgcStart+tgcSlope*currentDepth;
        tgcCurve.push_back(tgcValue);
        currentDepth += tgcSamplingStep;
    }
    return tgcCurve;
}

void setLinearTgc(Us4R *us4r) {
    try {
        float tgcStart, tgcSlope;
        std::cout << "TGC curve start value [dB]" << std::endl;
        std::cin >> tgcStart;
        std::cout << "TGC curve slope [dB/m]" << std::endl;
        std::cin >> tgcSlope;
        std::vector<float> tgcCurve = getLinearTGCCurve(
            tgcStart, tgcSlope, SAMPLING_FREQUENCY,
            SPEED_OF_SOUND, SAMPLE_RANGE_END);

        std::cout << "Applying TGC curve: " << std::endl;
        for(auto &value: tgcCurve) {
            std::cout << value << ", ";
        }
        std::cout << std::endl;
        us4r->setTgcCurve(tgcCurve);
    }
    catch(const arrus::IllegalArgumentException &e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
    }
}

int main() noexcept {
    try {
        // TODO set path to us4r-lite configuration file
        auto settings = ::arrus::io::readSessionSettings("C:/Users/Public/us4r.prototxt");
        auto session = ::arrus::session::createSession(settings);
        auto us4r = (::arrus::devices::Us4R *) session->getDevice("/Us4R:0");
        auto probe = us4r->getProbe(0);

        unsigned nElements = probe->getModel().getNumberOfElements().product();
        std::cout << "Probe with " << nElements << " elements." << std::endl;

        ::arrus::BitMask rxAperture(nElements, true);

        Pulse pulse(6e6, 2, false);
        ::std::pair<::arrus::uint32, arrus::uint32> sampleRange{0, SAMPLE_RANGE_END};

        std::vector<TxRx> txrxs;

		// plane waves
        for(int i = 0; i < 32; ++i) {
            // NOTE: the below vector should have size == probe number of elements.
            // This probably will be modified in the future
            // (delays only for active tx elements will be needed).
            std::vector<float> delays(nElements, 0.0f);
            for(int d = 0; d < nElements; ++d) {
                delays[d] = d*i*1e-9f;
            }
            arrus::BitMask txAperture(nElements, true);
            txrxs.emplace_back(Tx(txAperture, delays, pulse),
                               Rx(txAperture, sampleRange),
                               70e-6f);
        }

        TxRxSequence seq(txrxs, {});
        DataBufferSpec outputBuffer{DataBufferSpec::Type::FIFO, 12};
        std::cout << "ASYNC MODE" << std::endl;
        Scheme scheme(seq, 4, outputBuffer, Scheme::WorkMode::ASYNC);

        auto result = session->upload(scheme);
		us4r->setVoltage(5);

        std::condition_variable cv;
        using namespace std::chrono_literals;

        std::chrono::steady_clock::time_point last = std::chrono::steady_clock::now();

        OnNewDataCallback callback = [&, i = 0](const BufferElement::SharedHandle &ptr) mutable {
            try {
                if(i % 2000 == 0) {
                    std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now-last).count();
                    std::cout << "Alive " << i << ", elapsed time: " << duration << " [ms]" << std::endl;
                    last = now;
                }
                ptr->release();
                ++i;
            } catch(const std::exception &e) {
                std::cout << "Exception: " << e.what() << std::endl;
                cv.notify_all();
            } catch (...) {
                std::cout << "Unrecognized exception" << std::endl;
                cv.notify_all();
            }
        };

        OnOverflowCallback overflowCallback = [&] () {
            std::cout << "Data overflow occurred!" << std::endl;
            cv.notify_one();
        };

        // Register the callback for new data in the output buffer.
        auto buffer = std::static_pointer_cast<DataBuffer>(result.getBuffer());
        buffer->registerOnNewDataCallback(callback);
        buffer->registerOnOverflowCallback(overflowCallback);

        session->startScheme();

        char lastChar = 0;
        while (lastChar != 'q') {
            std::cout << "Menu: " << std::endl;
            std::cout << "v - set voltage" << std::endl;
            std::cout << "t - set linear tgc" << std::endl;
            std::cout << "q - quit" << std::endl;
            std::cout << "Choose an option and press enter" << std::endl;
            std::cin >> lastChar;
            switch(lastChar) {
                case 'v':
                    // Set voltage
                    setVoltage(us4r);
                    break;
                case 't':
                    setLinearTgc(us4r);
                    break;
                case 'q':
                    std::cout << "Stopping application" << std::endl;
                    break;
                default:
                    std::cerr << "Unknown command: " << lastChar << std::endl;
            }
        }
        // Stop the system.
        session->stopScheme();

    } catch(const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }

    return 0;
}
