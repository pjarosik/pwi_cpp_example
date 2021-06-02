#ifndef CPP_EXAMPLE_CFG_H
#define CPP_EXAMPLE_CFG_H

Session::Handle configureSessionUsingFile(const std::string &path) {
    auto settings = ::arrus::io::readSessionSettings(path);
    return ::arrus::session::createSession(settings);
}

::arrus::devices::ProbeAdapterSettings::ChannelMapping getEsaote3AdapterChannelMapping(::arrus::ChannelIdx nSystemChannels) {
    std::cout << "Using Esaote3 adapter-like channel mapping" << std::endl;
    ::arrus::devices::ProbeAdapterSettings::ChannelMapping mapping(nSystemChannels);
    // 1:1 channel mapping between Us4OEMs <-> Probe Adapter:
    // - Us4OEM:0 is connected to channels: [i*32, (i+1)*32), where "i" is even,
    // - Us4OEM:1 is connected to channels: [i*32, (i+1)*32), where "i" is odd,
    // i = 0, 1, ..., (N_CHANNELS / US4OEM_N_RX (32)) - 1

    // paCh means "Probe Adapter channel"
    // uCh means "Us4OEM channel"
    for(int paCh = 0; paCh < nSystemChannels; ++paCh) {
        unsigned i = paCh / 32;

        ::arrus::devices::ProbeAdapterSettings::Us4OEMOrdinal us4oem = i % 2;
        // The below means: RX GROUP number (nSystemChannels/64) * NRX_CHANNELS + channel number in that group
        // For example, probe adapter channel 34 will be translated to
        ::arrus::ChannelIdx uCh = i/2 * 32 + paCh%32;
        mapping[paCh] = ProbeAdapterSettings::ChannelAddress(us4oem, uCh);
    }
    return mapping;
}

::arrus::devices::ProbeAdapterSettings::ChannelMapping getUltrasonixChannelMapping(::arrus::ChannelIdx nSystemChannels) {
    std::cout << "Using Ultrasonix adapter channel mapping" << std::endl;
    std::vector<::arrus::ChannelIdx> channels = {
        // Us4OEM:0
        0,  1,  2,  3,  4,  5,  6,  7,
        8,  9,  10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23,
        24, 25, 26, 27, 28, 29, 30, 31,
        // Us4OEM:1
        63, 62, 61, 60, 59, 58, 57, 56,
        55, 54, 53, 52, 51, 50, 49, 48,
        47, 46, 45, 44, 43, 42, 41, 40,
        39, 38, 37, 36, 35, 34, 33, 32,
        // Us4OEM:0
        64, 65, 66, 67, 68, 69, 70, 71,
        72, 73, 74, 75, 76, 77, 78, 79,
        80, 81, 82, 83, 84, 85, 86, 87,
        88, 89, 90, 91, 92, 93, 94, 95,
        // Us4OEM:1
        127, 126, 125, 124, 123, 122, 121, 120,
        119, 118, 117, 116, 115, 114, 113, 112,
        111, 110, 109, 108, 107, 106, 105, 104,
        103, 102, 101, 100,  99,  98,  97,  96
    };
    std::vector<::arrus::devices::Ordinal> us4oems(nSystemChannels);
    std::fill(std::begin(us4oems), std::begin(us4oems)+32, 0);
    std::fill(std::begin(us4oems)+32, std::begin(us4oems)+64, 1);
    std::fill(std::begin(us4oems)+64, std::begin(us4oems)+96, 0);
    std::fill(std::begin(us4oems)+96, std::begin(us4oems)+128, 1);

    ::arrus::devices::ProbeAdapterSettings::ChannelMapping mapping;
    for(int i = 0; i < nSystemChannels; ++i) {
        mapping.push_back({us4oems[i], channels[i]});
    }
    return mapping;
}


Session::Handle configureCustomSession(::arrus::ChannelIdx nSystemChannels) {
    // --- Configuring probe adapter.
//    ::arrus::devices::ProbeAdapterSettings::ChannelMapping mapping = getEsaote3AdapterChannelMapping(nSystemChannels);
    ::arrus::devices::ProbeAdapterSettings::ChannelMapping mapping = getUltrasonixChannelMapping(nSystemChannels);

    ProbeAdapterSettings probeAdapterSettings {
        ProbeAdapterModelId{"acme", "mycustom"},
        (::arrus::ChannelIdx)mapping.size(),
        mapping
    };

    // Probe settings.
    ProbeModel probeModel {
        ProbeModelId{"ultrasonix", "l14-5/38"},
        ::arrus::Tuple<ProbeModel::ElementIdxType>({nSystemChannels}),
        // pitch
        ::arrus::Tuple<double>({0.3048e-3}),
        // The below sets a constraints on TX frequency values that can be set
        // using arrus libraries.
        ::arrus::Interval<float>(1e6, 20e6),
        // The below sets a constraints on voltage values that can be set using
        // arrus libraries.
        // NOTE: when using us4R-lite with HV256, max voltage is 90 V.
        ::arrus::Interval<::arrus::Voltage>(0, 75),
        // curvature radius, inf means flat linear array
        std::numeric_limits<double>::infinity()
    };
    // 1:1 channel mapping ProbeAdapter <-> Probe
    std::vector<::arrus::ChannelIdx> probe2AdapterMapping(nSystemChannels);
    std::generate(std::begin(probe2AdapterMapping),
                  std::end(probe2AdapterMapping),
                  [i = 0]() mutable {return i++;});

    ::arrus::devices::ProbeSettings probeSettings {
        probeModel,
        probe2AdapterMapping
    };
    // Default RX settings.
    ::arrus::devices::RxSettings rxSettings {
        24, // attenuation 24 [dB], maximum gain, constant
        // When using the analog TGC, please remember to turn off digital TGC by setting
        // std::nullopt a value above
        // Currently only the below PGA and LNA values are supported by arrus package
        30, 24, // PGA, LNA gain [dB]
        // No analog TGC (empty list means that the analog TGC will be turned off)
        {},
        // Note: make sure that appropriate cut-off is set appropriate for TX frequency band you use
        35000000,
        200
    };

    // HV settings. If you are not using HV256 (e.g. by using some external HV),
    // please set std::nullopt in us4RSettings.
    ::arrus::devices::HVSettings hvSettings {
        ::arrus::devices::HVModelId{"us4us", "hv256"}
    };

    ::arrus::devices::Us4RSettings us4RSettings{
        probeAdapterSettings,
        probeSettings,
        rxSettings,
        hvSettings,
        // No channel masking will be set (?)
        {}, // probe channel masking
        {{}, {}} // us4OEM channel masking, set separately for each Us4OEM
        // The above masking is used to turn off TX and RX on a selected
        // us4OEM channels for the whole session.
        // This can be useful when you know that some probe elements are
        // short-circuited and for safety they should not be firing.
        // Both of the above masks must complement each other - i.e.
        // probe's and us4oems mask must refer to the same us4OEM channels,
        // otherwise an exception will be thrown.
        // The above condition is in order to reduce the risk of setting
        // incorrect combination of adapter/channel masking.
    };
    ::arrus::session::SessionSettings sessionSettings{us4RSettings};
    return ::arrus::session::createSession(sessionSettings);
}
#endif //CPP_EXAMPLE_CFG_H
