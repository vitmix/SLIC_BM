#ifndef TESTER_H
#define TESTER_H

#include <opencv2/opencv.hpp>
#include <chrono>

#include "StereoAlgorithm.h"
#include "Utilities.h"

class Tester
{
public:
    Tester(const std::string & indir, const std::string & outdir, CalibrationParams & calib)
        : m_inDir{ indir }, m_outDir{ outdir }, m_calib{ calib }
    {
        groundTruth = utils::read_pfm_file(m_inDir + "disp0GT.pfm");
        nonOccludedMask = cv::imread(m_inDir + "mask0nocc.png", cv::IMREAD_GRAYSCALE);
        nonOccludedMask = nonOccludedMask == 255;
    }

    ~Tester()
    {
        if (out != nullptr)
            fclose(out);
    }

    void test(
        cv::Mat & im0,
        cv::Mat & im1,
        StereoAlgorithm & algo,
        const std::string & id,
        double errorThreshold = 0.5)
    {
        std::cout << "Start running " + algo.getName() + " algorithm\n";

        if (out == nullptr)
        {
            out = fopen((m_outDir + "log_out" + id + ".txt").c_str(), "w");
            if (out != nullptr)
            {
                fprintf(out, "\n%s\t%s\t%s\n", "Time", "All", "Non Occluded");
                fflush(out);
            }
        }

        auto start = std::chrono::high_resolution_clock::now();
        cv::Mat disparityMap;
        algo.match(im0, im1, disparityMap);
        auto end = std::chrono::high_resolution_clock::now();
        double timeTaken = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        timeTaken *= 1e-9; // seconds

        cv::Mat validMask = (groundTruth > 0.0) & (groundTruth != INFINITY);
        cv::Mat occludedMask = ~nonOccludedMask & validMask;

        int numOfValidPixels = cv::countNonZero(validMask);
        int numOfNonOccludedPixels = cv::countNonZero(nonOccludedMask);

        disparityMap.convertTo(disparityMap, groundTruth.type());
        cv::Mat errMap = cv::abs(disparityMap - groundTruth) <= errorThreshold;
        cv::Mat errMapVis = errMap | (~validMask);

        cv::imwrite(m_outDir + "ErrMap" + id + ".png", errMapVis);
        
        // ===================================================================================
        cv::Mat zmap0, zmap1;
        StereoAlgorithm::makeZMap(zmap0, disparityMap, m_calib);
        StereoAlgorithm::makeZMap(zmap1, groundTruth, m_calib);
        double all = 0.0, nonocc = 0.0;
        cv::Scalar zse = cv::sum(cv::abs(zmap1 - zmap0));
        std::cout << "\n" << zse << "\n";
        all = zse[0];
        // ===================================================================================

        if (out != nullptr)
        {
            fprintf(out, "%lf\t%lf\t%lf\n", timeTaken, all, nonocc);
            fflush(out);
        }
        utils::save_pfm_file(m_outDir + "disp" + id + ".pfm", disparityMap);
        std::cout << cv::format("%5.1lf\t%4.2lf\t%4.2lf", timeTaken, all, nonocc) << "\n";
        std::cout << "Stop running " + algo.getName() + " algorithm\n";
    }

private:
    cv::Mat groundTruth;
    cv::Mat nonOccludedMask;
    std::string m_inDir, m_outDir;
    CalibrationParams & m_calib;
    FILE * out = nullptr;
};

#endif