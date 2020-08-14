#ifndef STEREO_ALGORITHM_H
#define STEREO_ALGORITHM_H

#include <opencv2/opencv.hpp>
#include <unordered_map>
#include "Properties.h"
#include "SlicAlgorithm.h"
#include "WeightedGraph.h"


class StereoAlgorithm
{
public:
    virtual void match(
        cv::Mat & limg,
        cv::Mat & rimg,
        cv::Mat & dispmap) = 0;
    virtual ~StereoAlgorithm() = default;
    virtual std::string getName() const = 0;
    
    static void makeZMap(
        cv::Mat & zMap,
        cv::Mat & dispMap,
        CalibrationParams & calib);
};

class SlicBasedStereoM : public StereoAlgorithm
{
private:
    struct StatsParams
    {
        double sigma;
        double mean;

        void reset() { sigma = mean = 0.0; }
    };

    static inline double computeBhattaDistance(StatsParams & sp1, StatsParams & sp2);
    void computeStats(cv::Mat & img);
    void computeDistances(double sigma);

public:
    template <typename Key, typename Value>
    using umap = std::unordered_map<Key, Value>;

    SlicBasedStereoM(Properties & props);
    virtual ~SlicBasedStereoM() override = default;

    virtual void match(
        cv::Mat & limg,
        cv::Mat & rimg,
        cv::Mat & dispmap) override;

    template <typename T>
    static int maxTripleSum(std::vector<T> & rgbHist)
    {
        T currsum = rgbHist[0] + rgbHist[1];
        T maxsum  = currsum;
        size_t sz = rgbHist.size();
        
        for (size_t i = 1; i < sz - 1; i++)
        {
            currsum = rgbHist[i - 1] + rgbHist[i] + rgbHist[i + 1];

            if (maxsum < currsum)
                maxsum = currsum;
        }
        currsum = rgbHist[sz - 2] + rgbHist[sz - 1];
        return maxsum < currsum ? currsum : maxsum;
    }

    void setFileName(const std::string & name) {
        m_fileName = name;
    }

    virtual std::string getName() const override {
        return "Slic based stereo-matching";
    }

    std::vector<int> & getKLabels() {
        return m_klabels;
    }

    SlicAlgorithm & getSlic() {
        return m_slic;
    }

    //virtual void makeZMap(
    //    cv::Mat & zMap,
    //    cv::Mat & dispMap) override;

private:
    void performSlicSegmentation(cv::Mat & img);
    void performCostAggregation(cv::Mat & img, double sigma = 0.16);

    void brandNewCostComputation(cv::Mat & limg, cv::Mat & rimg);

    void makeDisparityMap(cv::Mat & dispMap);

private:
    SlicAlgorithm        m_slic;
    Properties &         m_props;
    cv::Mat              m_labeled;
    std::vector<int>     m_klabels, m_labelIDs;
    int                  m_width,   m_height;
    
    WeightedGraph                  m_graph;
    SlicAlgorithm::nmap            m_neimap;
    umap<int, int>                 m_labelsizes;
    umap<int, int>                 m_disparities;
    umap<int, StatsParams>         m_stats;
    umap<int, umap<int, double>>   m_costs, m_aggregated;

    std::string m_fileName;
};

class SemiGlobalBM : public StereoAlgorithm
{
public:
    SemiGlobalBM(Properties & props);
    virtual ~SemiGlobalBM() override = default;

    virtual void match(
        cv::Mat & limg,
        cv::Mat & rimg,
        cv::Mat & dispmap) override;

    virtual std::string getName() const override {
        return "Semi global stereo-matching";
    }

private:
    Properties m_props;
    int m_width, m_height;
};

#endif