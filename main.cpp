#include <iostream>
#include <iomanip>
#include <thread>
#include <fstream>
#include <cstdio>
#include <string>
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "arrus/core/api/arrus.h"

using namespace ::cv;
using namespace ::arrus::session;
using namespace ::arrus::devices;
using namespace ::arrus::ops::us4r;
using namespace ::arrus::framework;

#include "common.h"
// My custom logger, which I register in arrus.
#include "logging/MyCustomLoggerFactory.h"
#include "Display2D.h"
#include "imaging/Pipeline.cuh"

// Uncomment the below to save acquired channel RF data to 'rf.bin' file.
//#define DUMP_RF

constexpr float PI = 3.14159265f;
constexpr unsigned N_US4OEMS = 2;
constexpr unsigned US4OEM_N_RX = 32;
constexpr unsigned SYSTEM_N_RX = N_US4OEMS * US4OEM_N_RX;
constexpr unsigned N_PROBE_ELEMENTS = 128;

constexpr float SPEED_OF_SOUND = 1490;
// TX/RX parameters
constexpr unsigned N_ANGLES = 64;
// TX angles range
constexpr float MIN_ANGLE = -10.0f; // [deg]
constexpr float MAX_ANGLE = 10.0f; // [deg]
constexpr unsigned SAMPLE_RANGE_START =  0*1024;
constexpr unsigned SAMPLE_RANGE_END = 2*1024;
constexpr float TX_FREQUENCY = 6e6f; // [Hz]
constexpr float TX_N_PERIODS = 2; // number of cycles
constexpr unsigned DOWNSAMPLING_FACTOR = 1;
constexpr float SAMPLING_FREQUENCY = 65e6/DOWNSAMPLING_FACTOR; // [Hz]

constexpr float PRI = 100e-6; // [s]
// This is the time between consecutive sequence executions ("seuqence repetition interval").
// If the total PRI for a given sequence is smaller than SRI - the last TX/RX
// pri will be increased by SRI-sum(PRI)
constexpr float SRI = 20e-3; // [s]

constexpr unsigned N_SAMPLES = SAMPLE_RANGE_END-SAMPLE_RANGE_START;
// Use all probe elements for TX/RX
constexpr unsigned TX_RX_APERTURE_SIZE = N_PROBE_ELEMENTS;
// Number of Us4OEM output frames per each single probe's TX/RX (done by muxing RX channels).
// Example: rx aperture has with 192 elements, we have only 32 channels per us4OEM.
// Thus we get here 6 frames: 3 TX/RXs on the first Us4OEM module, 3 TX/RXs on the second one.
constexpr unsigned N_FRAMES_PER_ANGLE = TX_RX_APERTURE_SIZE / US4OEM_N_RX;
constexpr unsigned N_TXS_PER_ANGLE = TX_RX_APERTURE_SIZE / SYSTEM_N_RX;

// An example how to configure session (using a file or C++ API)
#include "cfg.h"

// An object representing window that displays the data.
Display2D mainDisplay;
// If true, the next frame will be
bool isLogTimestamps = false;

void setLinearTgc(Us4R *us4r);

void setVoltage(Us4R *us4r) {
    try {
        unsigned voltage = 5;
        std::cout << "Please provide the voltage to set [V]" << std::endl;
        std::cin >> voltage;
        us4r->setVoltage(voltage);
    } catch(const arrus::IllegalArgumentException& e) {
        std::cerr << e.what() << std::endl;
    }
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

// The below functions create PWI TX/RX sequence.
TxRxSequence createPwiSequence(const arrus::devices::ProbeModel &probeModel,
                               const std::vector<float> &angles) {
    // Apertures
    auto nElements = probeModel.getNumberOfElements()[0];
    if(nElements < TX_RX_APERTURE_SIZE) {
        throw std::runtime_error("Aperture size exceeds available number of probe elements.");
    }
    std::vector<bool> rxAperture(TX_RX_APERTURE_SIZE, true);
    std::vector<bool> txAperture(TX_RX_APERTURE_SIZE, true);

    // Delays
    std::vector<float> delays(nElements, 0.0f);
    Pulse pulse(TX_FREQUENCY, TX_N_PERIODS, false);
    ::std::pair<::arrus::uint32, arrus::uint32> sampleRange{
        SAMPLE_RANGE_START, SAMPLE_RANGE_END};
    std::vector<TxRx> txrxs;

    float pitch = probeModel.getPitch()[0];

    for (auto angle: angles) {
        std::vector<float> delays(nElements, 0.0f);

        // Compute array of TX delays.
        for (int i = 0; i < nElements; ++i) {
            delays[i] = pitch * i * sin(angle) / SPEED_OF_SOUND;
        }
        float minDelay = *std::min_element(std::begin(delays), std::end(delays));
        for (int i = 0; i < nElements; ++i) {
            delays[i] -= minDelay;
        }
        txrxs.emplace_back(Tx(txAperture, delays, pulse),
                           Rx(rxAperture, sampleRange),
                           PRI);
    }
    return TxRxSequence{txrxs, {}, SRI};
}

UploadResult uploadScheme(Session *session, const TxRxSequence &seq) {
    DataBufferSpec outputBuffer{DataBufferSpec::Type::FIFO, 4};
    Scheme scheme{seq, 2, outputBuffer, Scheme::WorkMode::HOST};
    return session->upload(scheme);
}

void logTimestamps(const int16_t* dataPtr) {
    // Timestamps for each TX/RX in the whole sequence are stored in
    // the frame metadata bytes.
    // The metadata bytes are available in the first 64 bytes of each frame
    // collected by Us4OEM:0.
    // Timestamp is stored in bytes 8:16, the value is in the number of clock
    // cycles, clock frequency: 65 MHz.
    // Here we are just printing out timestamps on console.

    std::cout << "Timestamps: " << std::endl;
    for(int frame = 0; frame < N_ANGLES * N_TXS_PER_ANGLE; ++frame) {
        char* framePtr = (char*)(dataPtr + frame*N_SAMPLES*US4OEM_N_RX);
        uint64_t timestampNCycles = *(uint64_t*)(framePtr+8);
        double timestamp = timestampNCycles / 65e6;
        std::cout << std::setprecision(10) << timestamp << ", ";
    }
    std::cout << std::endl;
}

void onProcessingEndCallback(void* input) {
    #ifdef DUMP_IMAGING_OUTPUT
        writeDataToFile("postprocessed.bin", (char*)input, 381*425*sizeof(uint8_t));
    #endif
    mainDisplay.update(input);
}

void registerProcessing(
        const std::shared_ptr<::arrus::framework::DataBuffer>& inputBuffer,
        std::shared_ptr<::imaging::Pipeline> processing,
        void(* onProcessingErrorCallback)(),
        void(* onRFDataBufferOverflowCallback)()) {

    auto releaseRFBufferElementFunc = [](void *element) {
        ((BufferElement*)element)->release();
    };

    OnNewDataCallback callback =
            [&, i = 0](const BufferElement::SharedHandle &ptr) mutable {
                try {
                    auto* dataPtr = ptr->getData().get<int16_t>();

                    if(isLogTimestamps) {
                        logTimestamps(dataPtr);
                        isLogTimestamps = false;
                    }
#ifdef DUMP_RF
                    writeDataToFile("rf.bin", (char*)dataPtr, ptr->getSize());
#endif
                    processing->process(dataPtr, onProcessingEndCallback, releaseRFBufferElementFunc, ptr.get());

                } catch (const std::exception &e) {
                    std::cout << "Exception: " << e.what() << std::endl;
                    onProcessingErrorCallback();
                } catch (...) {
                    std::cout << "Unrecognized exception" << std::endl;
                    onProcessingErrorCallback();
                }
            };
    inputBuffer->registerOnNewDataCallback(callback);
    // If using "ASYNC" mode, remember to register on overflow callback -
    // this function will be called when RF producer (us4oems) overrides RF
    // data buffer.
    // The registered overflow callback function doesn't matter for "HOST" mode.
    std::function<void()> bufferOverflowCallbackWrap = [&] () {
        onRFDataBufferOverflowCallback();
    };
    inputBuffer->registerOnOverflowCallback(bufferOverflowCallbackWrap);
}

auto copyChannelMappingData(const std::shared_ptr<FrameChannelMapping> &fcm) {
    auto nCh = fcm->getNumberOfLogicalChannels();
    auto nFr = fcm->getNumberOfLogicalFrames();

    // Actually, the below arrays represent 2D arrays with shape: (nFr, nCh).
    std::vector<int8_t> fcmChannels(nCh*nFr, -1);
    std::vector<uint16_t> fcmFrames(nCh*nFr, 0);

    // Iterate over logical frames and channels.
    for(uint16_t fr = 0; fr < nFr; ++fr) {
        for(uint16_t ch = 0; ch < nCh; ++ch) {
            auto [physicalFrame, physicalChannel] = fcm->getLogical(fr, ch);
            fcmFrames[fr*nCh+ch] = physicalFrame;
            fcmChannels[fr*nCh+ch] = physicalChannel;
        }
    }
    return std::make_pair(fcmChannels, fcmFrames);
}

int main() noexcept {
    try {
        // The below line register a custom logger in arrus package.
        // In order to get output for log messages with level < INFO, it is
        // necessary to register a custom logger factory. Please refer to the
        // MyCustomLoggerFactory implementation for more details.
        //
        // Also, please remember that ARRUS package only reports errors
        // by throwing exceptions, so it is therefore recommended to wrap
        // that uses ARRUS into try ..catch clauses.
        ::arrus::setLoggerFactory(std::make_shared<MyCustomLoggerFactory>(::arrus::LogSeverity::INFO));

//        auto session = configureSessionUsingFile("C:/Users/Public/us4r.prototxt");
        // Configure custom device. Please refer to cfg.h for more information.
        auto session = configureCustomSession(N_PROBE_ELEMENTS);
        auto us4r = (::arrus::devices::Us4R *) session->getDevice("/Us4R:0");
        auto probe = us4r->getProbe(0);

        // Creating TX/RX sequence to be executed by the device.
        auto txAngles = linspace(MIN_ANGLE*PI/180, MAX_ANGLE*PI/180, N_ANGLES);
        TxRxSequence seq = createPwiSequence(probe->getModel(), txAngles);

        // Upload TX/RX sequence on the device.
        auto result = uploadScheme(session.get(), seq);
        // Get upload results:
        // - RF buffer, which will be filled by Us4OEMS after the session is started.
        auto rfBuffer = std::static_pointer_cast<DataBuffer>(result.getBuffer());
        // - RF data description - currently contains only information about frame channel mapping.
        auto frameChannelMapping = result.getConstMetadata()->get<FrameChannelMapping>("frameChannelMapping");
        auto [fcmChannels, fcmFrames] = copyChannelMappingData(frameChannelMapping);

        // Create imaging pipeline, that will be used to reconstruct B-mode data
        auto imaging = ::imaging::createPwiImagingPipeline(
                {(N_ANGLES * N_FRAMES_PER_ANGLE) * N_SAMPLES, 32},
                fcmChannels.data(), fcmFrames.data(),
                txAngles.size(), probe->getModel().getNumberOfElements()[0],
                N_SAMPLES, SAMPLE_RANGE_START, txAngles,
                probe->getModel().getPitch()[0], SAMPLING_FREQUENCY, TX_FREQUENCY,
                TX_N_PERIODS, SPEED_OF_SOUND);

        // Update Displayed window dimensions according to imaging pipeline output.
        auto inputShape = imaging->getOutputShape();
        imaging::DataType inputDataType = imaging->getOutputDataType();
        if(inputShape.size() < 2) {
            throw std::runtime_error("Pipeline's output shape should have at "
                                     "least 2 dimensions.");
        }
        mainDisplay.setNrows(inputShape[inputShape.size()-2]);
        mainDisplay.setNcols(inputShape[inputShape.size()-1]);
        mainDisplay.setInputDataType(inputDataType);

        // Register processing pipeline for RF channel data buffer.
        registerProcessing(
                rfBuffer,
                imaging,
                []() {std::cout << "An error occurred while processing the data:" << std::endl; mainDisplay.exit();},
                [] () {std::cout << "RF data buffer overflow occurred. Stopping the system." << std::endl; mainDisplay.exit(); });

        us4r->setVoltage(20);
        session->startScheme();

        // Wait until the window is closed.
        // Stop the system.
        std::mutex mutex;
        std::unique_lock<std::mutex> lock(mutex);
        // Here the main thread waits until user presses 'q' button.
        // All the processing and displaying is done by callback threads.



        char lastChar = 0;
        while (lastChar != 'q') {
            std::cout << "Menu: " << std::endl;
            std::cout << "v - set voltage" << std::endl;
            std::cout << "t - set linear tgc" << std::endl;
            std::cout << "p - print timestamps" << std::endl;
            std::cout << "q - quit" << std::endl;
            std::cout << "Choose an option and press enter" << std::endl;
            lastChar = getchar();
            switch(lastChar) {
                case 'p':
                    // Set timestamp
                    isLogTimestamps = true;
                    break;
                case 'v':
                    // Set voltage
                    setVoltage(us4r);
                    break;
                case 't':
                    // Set TGC curve (linear)
                    setLinearTgc(us4r);
                    break;
                case 'q':
                    std::cout << "Stopping application" << std::endl;
                    break;
                default:
                    std::cerr << "Unknown command: " << lastChar << std::endl;
            }
        }
        mainDisplay.close();
        mainDisplay.waitUntilClosed(lock);

        session->stopScheme();
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }
    return 0;
}

