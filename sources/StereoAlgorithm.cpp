#include "StereoAlgorithm.h"
#include "SlicAlgorithm.h"
#include "Histogram.h"

#include <fstream>

SlicBasedStereoM::SlicBasedStereoM(Properties & props)
    : m_props{ props }, m_width{ 0 }, m_height{ 0 }
{}
    
void SlicBasedStereoM::match(
    cv::Mat & limg,
    cv::Mat & rimg,
    cv::Mat & dispmap)
{
    m_width  = limg.cols;
    m_height = limg.rows;

    if (rimg.type() != CV_32F)
        rimg.convertTo(rimg, CV_32F, 1.0 / 255.0);

    performSlicSegmentation(limg);
    brandNewCostComputation(limg, rimg);
    performCostAggregation(limg, m_props.sigma);
    makeDisparityMap(dispmap);
}

void SlicBasedStereoM::performSlicSegmentation(cv::Mat & img)
{
    m_slic.makeSegmentation(
        img, 
        m_labeled, 
        m_width, 
        m_height, 
        m_klabels, 
        m_labelIDs, 
        m_props.superpixsz,
        m_props.compactness,
        m_props.enforce);

    m_slic.findSpixNeighbors(m_neimap, m_labeled);
    //m_slic.makeContoursAroundSpixs(img, m_klabels, img.cols, img.rows, 0);
}

void SlicBasedStereoM::performCostAggregation(cv::Mat & img, double sigma)
{
    //std::cout << "Entering cost aggregation...\n";
    computeStats(img);
    //std::cout << "Stats was computed...\n";
    computeDistances(sigma);
    //std::cout << "Distances was computed...\n";

    int & minDisparity = m_props.calib.vmin;
    int & maxDisparity = m_props.calib.vmax;
    int dispRange = maxDisparity - minDisparity;
    m_aggregated.reserve(dispRange);
    
    for (int d = minDisparity; d <= maxDisparity; d++)
    {
        umap<int, double> aggCosts;
        double aggr = 0.0;

        for (auto & cost : m_costs[d])
        {
            auto & adjs = m_graph.getWeights(cost.first);
            aggr = cost.second;
        
            for (auto & adj : adjs) {
                aggr += adj.second * m_costs[d][adj.first];
            }
            aggCosts[cost.first] = aggr;
        }
        m_aggregated[d] = std::move(aggCosts);
    }

    // running Winner-Take-All
    umap<int, double> & currCost = m_costs.begin()->second;
    for (auto & lbl : currCost) { lbl.second = 0.0; }

    for (int d = minDisparity; d <= maxDisparity; d++)
    {
        for (auto & aggr : m_aggregated[d])
        {
            if (currCost[aggr.first] < aggr.second) {
                currCost[aggr.first] = aggr.second;
                m_disparities[aggr.first] = d;
            }
        }
    }
}

void SlicBasedStereoM::makeDisparityMap(cv::Mat & dispMap)
{
    
    //std::cout << "Computed old disparities (" << (m_props.calib.vmax - m_props.calib.vmin) << ") : \n";

    //for (auto & d : m_disparities) {
    //    std::cout << "[" << d.first << "; " << d.second << "]\n";
    //}

    int* srcptr = nullptr;
    uchar* dstptr = nullptr;
    dispMap = cv::Mat::zeros(m_height, m_width, CV_8UC1);

    for (int y = 0; y < m_height; y++)
    {
        srcptr = m_labeled.ptr<int>(y);
        dstptr = dispMap.ptr<uchar>(y);

        for (int x = 0; x < m_width; x++) {
            dstptr[x] = static_cast<uchar>(m_disparities[srcptr[x]]);
        }
    }
}

double SlicBasedStereoM::computeBhattaDistance(StatsParams & sp1, StatsParams & sp2)
{
    double firstTerm  = 0.25 * (sp1.sigma/sp2.sigma + sp2.sigma/sp1.sigma + 2);
    double secondTerm = ((sp1.mean - sp2.mean)*(sp1.mean - sp2.mean)) / (sp1.sigma + sp2.sigma);
    return (0.25 * std::log(firstTerm)) + (0.25 * secondTerm);
}

void SlicBasedStereoM::computeStats(cv::Mat & img)
{
    // compute stats of all super pixels
    cv::Vec3f *imptr = nullptr;
    int *lblptr = nullptr;

    for (auto & ID : m_labelIDs)
        m_stats[ID].reset();
    //-------------------------------------------
    // mean calculation
    for (int y = 0; y < img.rows; y++)
    {
        imptr  = img.ptr<cv::Vec3f>(y);
        lblptr = m_labeled.ptr<int>(y);
            
        for (int x = 0; x < img.cols; x++) {
            cv::Vec3f & pnt = imptr[x];
            m_stats[lblptr[x]].mean += pnt[0] + pnt[1] + pnt[2];
        }
    }
    for (auto & s : m_stats) {
        //s.second.mean /= static_cast<double>(3 * m_slic.getSpixSize(s.first));
        s.second.mean /= static_cast<double>(3 * m_labelsizes[s.first]);
    }
    //std::cout << "Mean was computed...\n";
    //-------------------------------------------
    // variance calculation
    for (int y = 0; y < img.rows; y++)
    {
        imptr = img.ptr<cv::Vec3f>(y);
        lblptr = m_labeled.ptr<int>(y);

        for (int x = 0; x < img.cols; x++)
        {
            double & mean = m_stats[lblptr[x]].mean;
            cv::Vec3f & pnt = imptr[x];
            m_stats[lblptr[x]].sigma += (pnt[0] - mean) * (pnt[0] - mean) +
                                        (pnt[1] - mean) * (pnt[1] - mean) +
                                        (pnt[2] - mean) * (pnt[2] - mean);
        }
    }
    for (auto & s : m_stats) {
        //s.second.sigma /= static_cast<double>(3 * m_slic.getSpixSize(s.first)) - 1.0;
        s.second.sigma /= static_cast<double>(3 * m_labelsizes[s.first]) - 1.0;
    }
    //std::cout << "Variance was computed...\n";
    //-------------------------------------------
}

void SlicBasedStereoM::computeDistances(double sigma)
{
    auto & stat = m_stats;
    m_graph.construct(m_neimap);
    m_graph.traverse([&stat, &sigma](int q, int p) -> double {
        double dist = computeBhattaDistance(stat[q], stat[p]);
        return std::exp((-1.0 * dist * dist) / (2 * sigma * sigma));
    });
    //std::cout << "\nResulted graph : \n" << m_graph << "\n\n";
}

void SlicBasedStereoM::brandNewCostComputation(cv::Mat & limg, cv::Mat & rimg)
{
    int & minDisparity = m_props.calib.vmin;
    int & maxDisparity = m_props.calib.vmax;
    int dispRange = maxDisparity - minDisparity;

    int *lblptr = nullptr;
    cv::Vec3f *limgptr = nullptr, *rimgptr = nullptr;

    umap<int, std::vector<double>> rComp, gComp, bComp;
    umap<int, int> idOfLastAdded;
    umap<int, double> cost;
    m_costs.reserve(dispRange);

    const int numOfBins = 20;
    const double low = m_props.lowBound, high = m_props.highBound;
    Histogram<double> hist(numOfBins, low, high);
    std::vector<int> rHist, gHist, bHist, accumHist;
    rHist.assign(numOfBins, 0);
    gHist.assign(numOfBins, 0);
    bHist.assign(numOfBins, 0);
    accumHist.assign(numOfBins, 0);

    // init label sizes
    for (auto & lblID : m_labelIDs) {
        m_labelsizes[lblID] = static_cast<int>(m_slic.getSpixSize(lblID));
    }
    // init (r,g,b) component maps
    for (auto & lbl : m_labelsizes)
    {
        rComp[lbl.first].assign(lbl.second, 0.0);
        gComp[lbl.first].assign(lbl.second, 0.0);
        bComp[lbl.first].assign(lbl.second, 0.0);
    }
    // main loop
    for (int d = minDisparity; d <= maxDisparity; d++)
    {
        lblptr = nullptr;
        limgptr = rimgptr = nullptr;

        for (auto & lblID : m_labelIDs) {
            idOfLastAdded[lblID] = 0;
        }

        for (int y = 0; y < m_height; y++)
        {
            lblptr = m_labeled.ptr<int>(y);
            limgptr = limg.ptr<cv::Vec3f>(y);
            rimgptr = rimg.ptr<cv::Vec3f>(y);

            for (int x = d; x < m_width; x++)
            //for (int x = 0; x < (m_width - d); x++)
            {
                int & lblID = lblptr[x];
                int & vecID = idOfLastAdded[lblID];
                cv::Vec3f & left = limgptr[x];
                cv::Vec3f & right = rimgptr[x - d];
                rComp[lblID][vecID] = left[0] / right[0];
                gComp[lblID][vecID] = left[1] / right[1];
                bComp[lblID][vecID] = left[2] / right[2];
                vecID++;
            }
        }

        for (auto & lblID : m_labelIDs)
        {
            for (int i = 0; i < numOfBins; i++) {
                rHist[i] = gHist[i] = bHist[i] = 0;
            }
            hist.calculateHist(rComp[lblID], rHist, m_fileName);
            hist.calculateHist(gComp[lblID], gHist, m_fileName);
            hist.calculateHist(bComp[lblID], bHist, m_fileName);

            for (int b = 0; b < numOfBins; b++) {
                accumHist[0] = rHist[b] + gHist[b] + bHist[b];
            }
            cost[lblID] = (1.0 / (3.0 * static_cast<double>(m_labelsizes[lblID]))) *
                (static_cast<double>(maxTripleSum(accumHist)));
        }
        m_costs[d] = std::move(cost);
    }
}

SemiGlobalBM::SemiGlobalBM(Properties & props)
    : m_props{ props }, m_width{ 0 }, m_height{ 0 }
{}

void SemiGlobalBM::match(cv::Mat & limg, cv::Mat & rimg, cv::Mat & dispmap)
{
    cv::Mat g1, g2;
    limg.convertTo(limg, CV_8U);
    rimg.convertTo(rimg, CV_8U);
    cv::cvtColor(limg, g1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(rimg, g2, cv::COLOR_BGR2GRAY);

    CalibrationParams & calib = m_props.calib;
    int dispRange = calib.vmax - calib.vmin;
    dispRange -= dispRange % 16;

    //cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(
    //    calib.vmin,
    //    dispRange,
    //    11,//3,
    //    100,//600,
    //    1000,//2400,
    //    32,//10,
    //    0,
    //    15,//1,
    //    1000,//150,
    //    16,//2,
    //    cv::StereoSGBM::MODE_HH
    //);
    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(
        calib.vmin,
        dispRange,
        3,
        600,
        2400,
        10,
        4,
        1,
        150,
        2,
        cv::StereoSGBM::MODE_HH
    );
    sgbm->compute(g1, g2, dispmap);
    cv::normalize(dispmap, dispmap, 0, 256, cv::NORM_MINMAX, CV_8U);
}

void StereoAlgorithm::makeZMap(cv::Mat & zMap, cv::Mat & dMap, CalibrationParams & calib)
{
    int width = dMap.cols, height = dMap.rows;
    int *dptr = nullptr;
    float* zptr = nullptr;
    zMap = cv::Mat::zeros(height, width, CV_32F);

    float f = calib.cam0[0][0];
    float B = calib.baseline;
    float cxL = calib.cam0[0][2];
    float cxR = calib.cam1[0][2];

    float cDiff = cxL - cxR;
    float fB = f * B, disp;

    for (int y = 0; y < height; y++)
    {
        zptr = zMap.ptr<float>(y);
        dptr = dMap.ptr<int>(y);

        for (int x = 0; x < width; x++) {
            //disp = dptr[x] == 0 ? 1.0 : static_cast<float>(dptr[x]);
            zptr[x] = fB / (dptr[x] - cDiff);
            zptr[x] = zptr[x] == 0.0 ? 1.0 : zptr[x];
        }
    }
}