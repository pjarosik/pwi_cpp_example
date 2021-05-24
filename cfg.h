#ifndef CPP_EXAMPLE_CFG_H
#define CPP_EXAMPLE_CFG_H

Session::Handle configureSessionUsingFile(const std::string &path) {
    auto settings = ::arrus::io::readSessionSettings(path);
    return ::arrus::session::createSession(settings);
}

Session::Handle configureCustomSession(::arrus::ChannelIdx nSystemChannels) {

    // --- Configuring probe adapter.
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
    ProbeAdapterSettings probeAdapterSettings {
        ProbeAdapterModelId{"us4us", "esaote3"},
        (::arrus::ChannelIdx)mapping.size(),
        mapping
    };

    // Probe settings.
    ProbeModel probeModel {
        ProbeModelId{"esaote", "sl1543"},
        ::arrus::Tuple<ProbeModel::ElementIdxType>({nSystemChannels}),
        // pitch
        ::arrus::Tuple<double>({0.245e-3}),
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
