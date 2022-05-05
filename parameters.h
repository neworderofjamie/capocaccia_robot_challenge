#pragma once

//------------------------------------------------------------------------
// Parameters
//------------------------------------------------------------------------
namespace Parameters
{
    enum DetectorAxis
    {
        DetectorAxisHorizontal,
        DetectorAxisMax,
    };

    // Order of detectors associated with each pixel
    enum Detector
    {
        DetectorLeft,
        DetectorRight,
        DetectorMax,
    };

    constexpr double timestep = 1.0;

    constexpr unsigned int inputWidth = 160;
	constexpr unsigned int inputHeight = 60;
    constexpr unsigned int kernelSize = 5;
    constexpr unsigned int centreWidth = 155;
	constexpr unsigned int centreHeight = 55;

    constexpr unsigned int macroPixelWidth = centreWidth / kernelSize;
	constexpr unsigned int macroPixelHeight = centreHeight / kernelSize;

    constexpr unsigned int detectorWidth = macroPixelWidth - 2;
	constexpr unsigned int detectorHeight = macroPixelHeight - 2;

    constexpr unsigned int outputScale = 12;
    constexpr unsigned int inputScale = 4;

    constexpr float flowPersistence = 0.94f;
    constexpr float spikePersistence = 0.97f;
    
    constexpr float outputVectorScale = 2.0f;
}
