// Standard C++ includes
#include <algorithm>
#include <fstream>
#include <iostream>
#include <mutex>
#include <random>
#include <thread>

// Standard C includes
#include <cassert>
#include <csignal>
#include <cstdlib>

// OpenCV includes
#include <opencv2/opencv.hpp>

// GeNN userproject includes
#include "timer.h"

// Common includes
#include "dvs.h"

// Model includes
#include "parameters.h"

// Auto-generated simulation code
#include "capocaccia_robot_challenge_CODE/definitions.h"

//----------------------------------------------------------------------------
// Anonymous namespace
//----------------------------------------------------------------------------
namespace
{
typedef void (*allocateFn)(unsigned int);

volatile std::sig_atomic_t g_SignalStatus;

void signalHandler(int status)
{
    g_SignalStatus = status;
}

unsigned int getNeuronIndex(unsigned int width, unsigned int x, unsigned int y)
{
    return x + (y * width);
}

void buildCentreToMacroConnection(unsigned int *rowLength, unsigned int *ind)
{
    // Calculate start and end of border on each row
    const unsigned int leftBorder = (Parameters::inputWidth - Parameters::centreWidth) / 2;
    const unsigned int rightBorder = leftBorder + Parameters::centreWidth;
	const unsigned int topBorder = (Parameters::inputHeight - Parameters::centreHeight) / 2;
    const unsigned int bottomBorder = topBorder + Parameters::centreHeight;

    // Loop through rows of pixels in centre
    unsigned int i = 0;
    for(unsigned int yi = 0; yi < Parameters::inputHeight; yi++){
        for(unsigned int xi = 0; xi < Parameters::inputWidth; xi++){
            // If we're in the centre
            if(xi >= leftBorder && xi < rightBorder && yi >= topBorder && yi < bottomBorder) {
                const unsigned int yj = (yi - topBorder) / Parameters::kernelSize;
                const unsigned int xj = (xi - leftBorder) / Parameters::kernelSize;
                
                ind[i] = getNeuronIndex(Parameters::macroPixelWidth, xj, yj);
                rowLength[i++] = 1;
            }
            else {
                rowLength[i++] = 0;
            }
        }
    }

    // Check
    assert(i == (Parameters::inputWidth * Parameters::inputHeight));
}

void buildDetectors(unsigned int *excitatoryRowLength, unsigned int *excitatoryInd,
                    unsigned int *inhibitoryRowLength, unsigned int *inhibitoryInd)
{
    // Loop through macro cells
    unsigned int iExcitatory = 0;
    unsigned int iInhibitory = 0;
    for(unsigned int yi = 0; yi < Parameters::macroPixelHeight; yi++){
        for(unsigned int xi = 0; xi < Parameters::macroPixelWidth; xi++){
            // Get index of start of row
            unsigned int sExcitatory = (iExcitatory * Parameters::DetectorMax);
            unsigned int sInhibitory = (iInhibitory * Parameters::DetectorMax);
            
            // If we're not in border region
            if(xi >= 1 && xi < (Parameters::macroPixelWidth - 1)
                && yi >= 1 && yi < (Parameters::macroPixelHeight - 1))
            {
                const unsigned int xj = (xi - 1) * Parameters::DetectorMax;
                const unsigned int yj = yi - 1;

                // Add excitatory synapses to all detectors
                for(unsigned int d = 0; d < Parameters::DetectorMax; d++) {
                    excitatoryInd[sExcitatory++] = getNeuronIndex(Parameters::detectorWidth * Parameters::DetectorMax,
                                                                  xj + d, yj);
                }
                excitatoryRowLength[iExcitatory++] = Parameters::DetectorMax;
            }
            else {
                excitatoryRowLength[iExcitatory++] = 0;
            }


            // Create inhibitory connection to 'left' detector associated with macropixel one to right
            inhibitoryRowLength[iInhibitory] = 0;
            if(xi < (Parameters::macroPixelWidth - 2)
                && yi >= 1 && yi < (Parameters::macroPixelHeight - 1))
            {
                const unsigned int xj = (xi - 1 + 1) * Parameters::DetectorMax;
                const unsigned int yj = yi - 1;
                inhibitoryInd[sInhibitory++] = getNeuronIndex(Parameters::detectorWidth * Parameters::DetectorMax,
                                                              xj + Parameters::DetectorLeft, yj);
                inhibitoryRowLength[iInhibitory]++;
            }

            // Create inhibitory connection to 'right' detector associated with macropixel one to right
            if(xi >= 2
                && yi >= 1 && yi < (Parameters::macroPixelHeight - 1))
            {
                const unsigned int xj = (xi - 1 - 1) * Parameters::DetectorMax;
                const unsigned int yj = yi - 1;
                inhibitoryInd[sInhibitory++] = getNeuronIndex(Parameters::detectorWidth * Parameters::DetectorMax,
                                                              xj + Parameters::DetectorRight, yj);
                inhibitoryRowLength[iInhibitory]++;
            }

            // Create inhibitory connection to 'up' detector associated with macropixel one below
            /*if(xi >= 1 && xi < (Parameters::macroPixelWidth - 1)
                && yi < (Parameters::macroPixelHeight - 2))
            {
                const unsigned int xj = (xi - 1) * Parameters::DetectorMax;
                const unsigned int yj = yi - 1 + 1;
                inhibitoryInd[sInhibitory++] = getNeuronIndex(Parameters::detectorWidth * Parameters::DetectorMax,
                                                              xj + Parameters::DetectorUp, yj);
                inhibitoryRowLength[iInhibitory]++;
            }

            // Create inhibitory connection to 'down' detector associated with macropixel one above
            if(xi >= 1 && xi < (Parameters::macroPixelWidth - 1)
                && yi >= 2)
            {
                const unsigned int xj = (xi - 1) * Parameters::DetectorMax;
                const unsigned int yj = yi - 1 - 1;
                inhibitoryInd[sInhibitory++] = getNeuronIndex(Parameters::detectorSize * Parameters::DetectorMax,
                                                              xj + Parameters::DetectorDown, yj);
                inhibitoryRowLength[iInhibitory]++;
            }*/
            iInhibitory++;

        }
    }

    // Check
    assert(iExcitatory == (Parameters::macroPixelWidth * Parameters::macroPixelHeight));
    assert(iInhibitory == (Parameters::macroPixelWidth * Parameters::macroPixelHeight));
}

void displayThreadHandler(std::mutex &inputMutex, const cv::Mat &inputImage, std::mutex &outputMutex, 
                          const float (&output)[Parameters::detectorWidth][Parameters::detectorHeight][Parameters::DetectorAxisMax])
{
    cv::namedWindow("Input", cv::WINDOW_NORMAL);
    cv::resizeWindow("Input", Parameters::inputWidth * Parameters::inputScale,
                     Parameters::inputHeight * Parameters::inputScale);

    // Create output image
    const unsigned int outputImageWidth = Parameters::detectorWidth * Parameters::outputScale;
	const unsigned int outputImageHeight = Parameters::detectorHeight * Parameters::outputScale;
    cv::Mat outputImage(outputImageHeight, outputImageWidth, CV_8UC3);

#ifdef JETSON_POWER
    std::ifstream powerStream("/sys/devices/platform/7000c400.i2c/i2c-1/1-0040/iio_device/in_power0_input");
    std::ifstream gpuPowerStream("/sys/devices/platform/7000c400.i2c/i2c-1/1-0040/iio_device/in_power1_input");
    std::ifstream cpuPowerStream("/sys/devices/platform/7000c400.i2c/i2c-1/1-0040/iio_device/in_power2_input");
#endif  // JETSON_POWER

    while(g_SignalStatus == 0){
        // Clear background
        outputImage.setTo(cv::Scalar::all(0));

        {
            std::lock_guard<std::mutex> lock(outputMutex);

            // Loop through output coordinates
            for(unsigned int x = 0; x < Parameters::detectorWidth; x++){
                for(unsigned int y = 0; y < Parameters::detectorHeight; y++){
                    const cv::Point start(x * Parameters::outputScale, y * Parameters::outputScale);
                    //const cv::Point end = start + cv::Point(Parameters::outputVectorScale * output[x][y][0],
                    //                                        Parameters::outputVectorScale * output[x][y][1]);
					const cv::Point end = start + cv::Point(Parameters::outputVectorScale * output[x][y][0], 0);
                    cv::line(outputImage, start, end,
                             CV_RGB(0xFF, 0xFF, 0xFF));
                }
            }
        }


#ifdef JETSON_POWER
        // Read all power measurements
        unsigned int power, cpuPower, gpuPower;
        powerStream >> power;
        cpuPowerStream >> cpuPower;
        gpuPowerStream >> gpuPower;

        // Clear all stream flags (EOF gets set)
        powerStream.clear();
        cpuPowerStream.clear();
        gpuPowerStream.clear();

        char power[255];
        sprintf(power, "Power:%umW, GPU power:%umW", power, gpuPower);
        cv::putText(outputImage, power, cv::Point(0, outputImageHeight - 20),
                    cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(0, 0, 0xFF));
        sprintf(power, "CPU power:%umW", cpuPower);
        cv::putText(outputImage, power, cv::Point(0, outputImageHeight - 5),
                    cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(0, 0, 0xFF));
#endif

        cv::imshow("Output", outputImage);

        {
            std::lock_guard<std::mutex> lock(inputMutex);
            cv::imshow("Input", inputImage);
        }


        cv::waitKey(33);
    }
}

void applyOutputSpikes(unsigned int outputSpikeCount, const unsigned int *outputSpikes, 
                       float (&output)[Parameters::detectorWidth][Parameters::detectorHeight][Parameters::DetectorAxisMax])
{
    // Loop through output spikes
    for(unsigned int s = 0; s < outputSpikeCount; s++)
    {
        // Convert spike ID to x, y, detector
        const unsigned int spike = outputSpikes[s];
        const auto spikeCoord = std::div((int)spike, (int)Parameters::detectorWidth * Parameters::DetectorMax);
        const int spikeY = spikeCoord.quot;
        const auto xCoord = std::div(spikeCoord.rem, (int)Parameters::DetectorMax);
        const int spikeX =  xCoord.quot;

        // Apply spike to correct axis of output pixel based on detector it was emitted by
        switch(xCoord.rem) {
            case Parameters::DetectorLeft:
                output[spikeX][spikeY][0] -= 1.0f;
                break;

            case Parameters::DetectorRight:
                output[spikeX][spikeY][0] += 1.0f;
                break;

            /*case Parameters::DetectorUp:
                output[spikeX][spikeY][1] -= 1.0f;
                break;

            case Parameters::DetectorDown:
                output[spikeX][spikeY][1] += 1.0f;
                break;*/

        }
    }

    // Decay output
    for(unsigned int x = 0; x < Parameters::detectorWidth; x++) {
        for(unsigned int y = 0; y < Parameters::detectorHeight; y++){
            for(unsigned int d = 0; d < Parameters::DetectorAxisMax; d++) {
                output[x][y][d] *= Parameters::flowPersistence;
            }
        } 
    }
}
}

int main()
{
    constexpr unsigned int timestepWords = ((Parameters::inputWidth * Parameters::inputHeight) + 31) / 32;
    
    allocateMem();
    allocatespikeVectorDVS(timestepWords);
    initialize();

    buildCentreToMacroConnection(rowLengthDVS_MacroPixel, indDVS_MacroPixel);
    buildDetectors(rowLengthMacroPixel_Flow_Excitatory, indMacroPixel_Flow_Excitatory,
                   rowLengthMacroPixel_Flow_Inhibitory, indMacroPixel_Flow_Inhibitory);

    initializeSparse();

    // Filter to extract positive events in bottom half of visual field
    using Filter = DVS::CombineFilter<DVS::PolarityFilter<DVS::Polarity::ON>, DVS::ROIFilter<0, 640, 120, 480>>;

    // Transform X coordinates by multiplying by 0.25
    using TransformX = DVS::Scale<8192>;

    // Transform Y coordinates by subtracting 240 and multiplying result by 0.25
    using TransformY = DVS::CombineTransform<DVS::Subtract<120>, DVS::Scale<8192>>;

    // Create DVXplorer device
    DVS::DVXplorer dvs;
    dvs.start();
    
    double dvsGet = 0.0;
    double step = 0.0;
    double render = 0.0;

    std::mutex inputMutex;
    cv::Mat inputImage(Parameters::inputHeight, Parameters::inputWidth, CV_32F);

    std::mutex outputMutex;
    float output[Parameters::detectorWidth][Parameters::detectorHeight][Parameters::DetectorAxisMax] = {0};
    std::thread displayThread(displayThreadHandler,
                              std::ref(inputMutex), std::ref(inputImage),
                              std::ref(outputMutex), std::ref(output));

    // Convert timestep to a duration
    const auto dtDuration = std::chrono::duration<double, std::milli>{DT};

    // Duration counters
    std::chrono::duration<double> sleepTime{0};
    std::chrono::duration<double> overrunTime{0};
    unsigned int i = 0;
    
    // Catch interrupt (ctrl-c) signals
    std::signal(SIGINT, signalHandler);

    for(i = 0; g_SignalStatus == 0; i++)
    {
        auto tickStart = std::chrono::high_resolution_clock::now();

        {
            TimerAccumulate timer(dvsGet);
            
            std::fill_n(spikeVectorDVS, timestepWords, 0);
            dvs.readEvents<Parameters::inputWidth, Filter, TransformX, TransformY>(spikeVectorDVS);

            // Copy to GPU
            pushspikeVectorDVSToDevice(timestepWords);
        }

        {
            TimerAccumulate timer(render);
            std::lock_guard<std::mutex> lock(inputMutex);

            {
                for(unsigned int w = 0; w < timestepWords; w++) {
                    // Get word
                    uint32_t spikeWord = spikeVectorDVS[w];
                    
                    // Calculate neuron id of highest bit of this word
                    unsigned int neuronID = (w * 32) + 31;
                    
                    // While bits remain
                    while(spikeWord != 0) {
                        // Calculate leading zeros
                        const int numLZ = __builtin_clz(spikeWord);
                        
                        // If all bits have now been processed, zero spike word
                        // Otherwise shift past the spike we have found
                        spikeWord = (numLZ == 31) ? 0 : (spikeWord << (numLZ + 1));
                        
                        // Subtract number of leading zeros from neuron ID
                        neuronID -= numLZ;
                        
                        // Write out CSV line
                        const auto spikeCoord = std::div((int)neuronID, (int)Parameters::inputWidth);
                        inputImage.at<float>(spikeCoord.quot, spikeCoord.rem) += 1.0f;
                        
                        // New neuron id of the highest bit of this word
                        neuronID--;
                    }
                }

                // Decay image
                inputImage *= Parameters::spikePersistence;
            }
        }

        {
            TimerAccumulate timer(step);

            // Simulate
            stepTime();
            pullFlowCurrentSpikesFromDevice();
        }

        {
            TimerAccumulate timer(render);
            {
                // **TODO** use spikerecording with 1 timestep buffer
                std::lock_guard<std::mutex> lock(outputMutex);
                applyOutputSpikes(spikeCount_Flow, spike_Flow, output);
            }
        }

        // Get time of tick start
        auto tickEnd = std::chrono::high_resolution_clock::now();

        // If there we're ahead of real-time pause
        auto tickDuration = tickEnd - tickStart;
        if(tickDuration < dtDuration) {
            auto tickSleep = dtDuration - tickDuration;
            sleepTime += tickSleep;
            std::this_thread::sleep_for(tickSleep);
        }
        else {
            overrunTime += (tickDuration - dtDuration);
        }
    }

    // Wait for display thread to die
    displayThread.join();

    std::cout << "Ran for " << i << " " << DT << "ms timesteps, overan for " << overrunTime.count() << "s, slept for " << sleepTime.count() << "s" << std::endl;
    std::cout << "Average DVS:" << (dvsGet * 1000.0) / i<< "ms, Step:" << (step * 1000.0) / i << "s, Render:" << (render * 1000.0) / i<< std::endl;

    return 0;
}
