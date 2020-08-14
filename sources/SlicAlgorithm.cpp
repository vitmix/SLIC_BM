#include "SlicAlgorithm.h"
#include <algorithm>

SlicAlgorithm::SlicAlgorithm()
    : m_width{ 0 }, m_height{ 0 }
{}

void SlicAlgorithm::makeSegmentation(
    cv::Mat &          img,
    cv::Mat &          labeled,
    const int &        width,
    const int &        height,
    std::vector<int> & klabels,
    std::vector<int> & labelids,
    const int &        spixsize,
    const double &     compactness,
    bool               enf)
{
    int sz = width * height;
    const int step = static_cast<int>(std::sqrt(static_cast<double>(spixsize)) + 0.5);
    std::vector<Seed> kseeds;

    m_width = width;
    m_height = height;
    std::vector<int> initlables(sz, -1);
    klabels.assign(sz, -1);

    if (img.type() != CV_32F)
        img.convertTo(img, CV_32F, 1.0 / 255.0);
    cv::cvtColor(img, labimg, cv::COLOR_BGR2Lab);

    initSeedsOfSpixs(kseeds, step);
    performSegmentation(kseeds, initlables, step, compactness);

    labeled = cv::Mat(m_height, m_width, CV_32S);
    if (enf) {
        enforce(initlables, klabels, static_cast<double>(sz) / static_cast<double>(step*step));
        constructSpxMat(klabels, m_width, m_height);
        std::memcpy(labeled.data, klabels.data(), klabels.size() * sizeof(int));
    }
    else {
        constructSpxMat(initlables, m_width, m_height);
        std::memcpy(labeled.data, initlables.data(), initlables.size() * sizeof(int));
        klabels = std::move(initlables);
    }
    labelids.reserve(spxmat.size());
    for (const auto & spx : spxmat) { labelids.push_back(spx.first); }
    std::sort(labelids.begin(), labelids.end());
}

void SlicAlgorithm::makeContoursAroundSpixs(
    cv::Mat &            img,
    std::vector<int> &   labels,
    const int &          width,
    const int &          height,
    const unsigned int & color)
{
    //=================================================================

    const int dx[8] = { -1, -1,  0,  1, 1, 1, 0, -1 };
    const int dy[8] = { 0, -1, -1, -1, 0, 1, 1,  1 };
    
    int sz = width * height;
    std::vector<bool> istaken(sz, false);
    std::vector<int>  contourx(sz);
    std::vector<int>  contoury(sz);
    
    int mainindex = 0, cind = 0;
    
    for (int j = 0; j < height; j++)
    {
        for (int k = 0; k < width; k++)
        {
            int np = 0;
            for (int i = 0; i < 8; i++)
            {
                int x = k + dx[i];
                int y = j + dy[i];
    
                if ((x >= 0 && x < width) &&
                    (y >= 0 && y < height))
                {
                    //if (false == istaken[y * width + x])
                        if (labels[mainindex] != labels[y * width + x]) np++;
                }
            }
    
            if (np > 1)
            {
                contourx[cind] = k;
                contoury[cind] = j;
                istaken[mainindex] = true;
                cind++;
            }
            mainindex++;
        }
    }
    
    int numboundpix = cind;
    for (int j = 0; j < numboundpix; j++)
    {
        //auto & pixel_w = img.at<cv::Vec3i>(contoury[j], contourx[j]);
        //pixel_w[0] = pixel_w[1] = pixel_w[2] = 1;
        //img.at<int>(contoury[j], contourx[j]) = 0xffffff;

        for (int n = 0; n < 8; n++)
        {
            int x = contourx[j] + dx[n];
            int y = contoury[j] + dy[n];
    
            if ((x >= 0 && x < width) && (y >= 0 && y < height))
                if (!istaken[y * width + x])
                {
                    //img.at<float>(y, x) = 0;
                    auto & pixel_b = img.at<cv::Vec3i>(y, x);
                    //auto & pixel_b = img.at<cv::Vec3f>(y, x);
                    pixel_b[0] = pixel_b[1] = pixel_b[2] = 0;
                }
        }
    }
}

void SlicAlgorithm::makeContours(
    cv::Mat &          img, 
    std::vector<int> & labels, 
    const int &        width, 
    const int &        height)
{
    if (spxmat.empty())
        constructSpxMat(labels, m_width, m_height);

    int maxx = 0, maxy = 0;
    cv::Mat zeros = cv::Mat::ones(img.size(), CV_32SC1);

    for (auto & spx : spxmat)
    {
        // while y is not changed iterate
        // when y changes - set zero
        int prevy = spx.second.front().y;
        for (auto & lbl : spx.second)
        {
            if (prevy != lbl.y)
            {
                auto & pixel = img.at<cv::Vec3i>(lbl);
                pixel[0] = pixel[1] = pixel[2] = 0;
                //img.at<float>(lbl.y, lbl.x) = 0;
                prevy = lbl.y;
            }
        }
    }
}

void SlicAlgorithm::performSegmentation(
    std::vector<Seed> & kseeds,
    std::vector<int> &  klabels,
    const int &         step,
    const double &      m /*= 10.0 */)
{
    const int sz = m_width * m_height;
    const int numseeds = static_cast<int>(kseeds.size());

    std::vector<double> clustersizes(numseeds, 0.0);
    std::vector<double> invs(numseeds, 0.0);
    std::vector<double> distances(sz, DBL_MAX);
    std::vector<Seed>   sigma(numseeds);

    double invwt = 1.0 / ((step / m)*(step / m));

    int        x0{ 0 }, x1{ 0 }, y0{ 0 }, y1{ 0 }, idx{ 0 };
    double     distlab{ 0.0 }, distxy{ 0.0 };
    cv::Vec3f* rowptr = nullptr;
    cv::Vec3f  pnt;

    for (int iter = 0; iter < 10; iter++)
    {
        distances.assign(sz, DBL_MAX);
        for (int n = 0; n < numseeds; n++)
        {
            Seed & seed = kseeds[n];

            x0 = std::max(0.0, seed.x - step);
            y0 = std::max(0.0, seed.y - step);
            x1 = std::min(static_cast<double>(m_width), seed.x + step);
            y1 = std::min(static_cast<double>(m_height), seed.y + step);

            for (int y = y0; y < y1; y++)
            {
                rowptr = labimg.ptr<cv::Vec3f>(y);
                idx = y * m_width;
                for (int x = x0; x < x1; x++)
                {
                    pnt = rowptr[x];

                    distlab = (pnt[0] - seed.L)*(pnt[0] - seed.L) +
                        (pnt[1] - seed.A)*(pnt[1] - seed.A) +
                        (pnt[2] - seed.B)*(pnt[2] - seed.B);

                    distxy = (x - seed.x)*(x - seed.x) +
                             (y - seed.y)*(y - seed.y);

                    //distlab += distxy * invwt;
                    distlab = std::sqrt(distlab) + std::sqrt(distxy * invwt);

                    // idx = y * m_width + x;
                    if (distlab < distances[idx + x])
                    {
                        distances[idx + x] = distlab;
                        klabels[idx + x] = n;
                    }
                }
            }
        }

        for (auto & seed : sigma) seed.reset();
        clustersizes.assign(numseeds, 0.0);

        idx = 0;
        for (y0 = 0; y0 < m_height; y0++)
        {
            rowptr = labimg.ptr<cv::Vec3f>(y0);
            for (x0 = 0; x0 < m_width; x0++)
            {
                //if (klabels[idx] != -1)
                //{
                    pnt = rowptr[x0];
                    Seed & seed = sigma[klabels[idx]];
                    seed.L += pnt[0];
                    seed.A += pnt[1];
                    seed.B += pnt[2];
                    seed.x += x0;
                    seed.y += y0;

                    clustersizes[klabels[idx]] += 1.0;
                //}
                idx++;
            }
        }

        for (idx = 0; idx < numseeds; idx++)
        {
            if (clustersizes[idx] <= 0) clustersizes[idx] = 1;
            invs[idx] = 1.0 / clustersizes[idx];
        }

        for (idx = 0; idx < numseeds; idx++)
        {
            double & inv = invs[idx];
            Seed   & oldseed = kseeds[idx];
            Seed   & newseed = sigma[idx];

            oldseed.L = newseed.L * inv;
            oldseed.A = newseed.A * inv;
            oldseed.B = newseed.B * inv;
            oldseed.x = newseed.x * inv;
            oldseed.y = newseed.y * inv;
        }
    }
}

void SlicAlgorithm::initSeedsOfSpixs(
    std::vector<Seed> & kseeds,
    const int &         step)
{
    int numseeds{ 0 };

    int xstrips = (0.5 + static_cast<double>(m_width) / static_cast<double>(step));
    int ystrips = (0.5 + static_cast<double>(m_height) / static_cast<double>(step));

    int xerr = m_width - xstrips * step;
    int yerr = m_height - ystrips * step;

    if (xerr < 0) { xstrips--; xerr = m_width - xstrips * step; }
    if (yerr < 0) { ystrips--; yerr = m_height - ystrips * step; }

    double xerrPerStrip = static_cast<double>(xerr) / static_cast<double>(xstrips);
    double yerrPerStrip = static_cast<double>(yerr) / static_cast<double>(ystrips);

    int xoffset{ 0 }, yoffset{ 0 };
    xoffset = yoffset = step / 2;

    numseeds = xstrips * ystrips;

    kseeds.resize(numseeds);

    int n{ 0 }, yseed{ 0 }, xseed{ 0 };
    cv::Vec3f* rowptr = nullptr;
    cv::Vec3f pnt;

    for (int y = 0; y < ystrips; y++)
    {
        yseed = (y * step) + yoffset + (y * yerrPerStrip);
        rowptr = labimg.ptr<cv::Vec3f>(yseed);

        for (int x = 0; x < xstrips; x++)
        {
            xseed = x * step + xoffset + (x * xerrPerStrip);
            pnt = rowptr[xseed];

            Seed & seed = kseeds[n];
            seed.L = pnt[0];
            seed.A = pnt[1];
            seed.B = pnt[2];
            seed.x = xseed;
            seed.y = yseed;
            n++;
        }
    }
}

void SlicAlgorithm::enforceLabelConnectivity(
    std::vector<int> & klabeled,
    std::vector<int> & labelids,
    const int &        spixsize)
{
    if (!spxmat.empty() && spixsize > 0)
    {
        std::vector<int> lblIdToMerge;

        for (auto & spx : spxmat)
        {
            if (spx.second.size() < spixsize && spx.second.size() > 0)
                lblIdToMerge.push_back(spx.first);
        }

        if (!lblIdToMerge.empty())
        {
            const int dx[4] = { -1, -0, 1, 0 };
            const int dy[4] = { 0, -1, 0, 1 };

            int x0 = 0, y0 = 0, x1 = 0, y1 = 0;
            int maxspix0 = -1, maxspix1 = -1, currid = 0;
            size_t spixsz0 = spixsize, spixsz1 = spixsize;

            for (auto & lblid : lblIdToMerge)
            {
                std::vector<cv::Point2i> & pixToMerge = spxmat[lblid];
                cv::Point2i & leftmostpnt = pixToMerge.front();
                cv::Point2i & rightmostpnt = pixToMerge.back();

                maxspix0 = maxspix1 = -1;
                spixsz0 = spixsz1 = spixsize;

                // finding spix with max size
                for (int n = 0; n < 4; n++)
                {
                    x0 = leftmostpnt.x + dx[n];
                    y0 = leftmostpnt.y + dy[n];

                    x1 = rightmostpnt.x + dx[n];
                    y1 = rightmostpnt.y + dy[n];

                    if ((x0 >= 0 && x0 < m_width) &&
                        (y0 >= 0 && y0 < m_height))
                    {
                        currid = klabeled[y0 * m_width + x0];
                        if (spxmat[currid].size() >= spixsz0)
                        {
                            spixsz0 = spxmat[currid].size();
                            maxspix0 = currid;
                        }
                    }

                    if ((x1 >= 0 && x1 < m_width) &&
                        (y1 >= 0 && y1 < m_height))
                    {
                        currid = klabeled[y1 * m_width + x1];
                        if (spxmat[currid].size() >= spixsz1)
                        {
                            spixsz1 = spxmat[currid].size();
                            maxspix1 = currid;
                        }
                    }
                }

                maxspix0 = std::max(maxspix0, maxspix1);
                if (maxspix0 > -1)
                {
                    // maxspixid0 --> label to merge with
                    std::vector<cv::Point2i> & pixToMergeWith = spxmat[maxspix0];
                    setSpxToValue<int>(lblid, klabeled, m_width, maxspix0);
                    pixToMergeWith.reserve(pixToMergeWith.size() + pixToMerge.size());

                    for (int i = 0; i < pixToMerge.size(); i++)
                        pixToMergeWith.push_back(std::move(pixToMerge[i]));

                    spxmat.erase(lblid);

                    std::sort(pixToMergeWith.begin(), pixToMergeWith.end(),
                        [](const cv::Point2i & a, const cv::Point2i & b) -> bool {
                        return a.y < b.y ? true : a.y == b.y ? a.x <= b.x : false;
                    });
                }
            }
        }
    
        labelids.reserve(spxmat.size());
        for (const auto & spx : spxmat)
            labelids.push_back(spx.first);

        std::sort(labelids.begin(), labelids.end());
    }
}

void SlicAlgorithm::enforce(
    std::vector<int> & oldlabels, 
    std::vector<int> & newlabels, 
    const int &        K)
{
    const int dx[4] = { -1,  0,  1,  0 };
    const int dy[4] = { 0, -1,  0,  1 };

    const int sz = m_width * m_height;
    const int spixSz = sz / K;

    //newlabels.assign(sz, -1);

    int label = 0, oindex = 0, adjlabel = 0;
    std::vector<int> xvec(sz, 0);
    std::vector<int> yvec(sz, 0);

    for (int j = 0; j < m_height; j++)
    {
        for (int k = 0; k < m_width; k++)
        {
            if (newlabels[oindex] < 0)
            {
                newlabels[oindex] = label;
                xvec[0] = k;
                yvec[0] = j;

                {for (int n = 0; n < 4; n++)
                {
                    int x = xvec[0] + dx[n];
                    int y = yvec[0] + dy[n];
                    if ((x >= 0 && x < m_width) && (y >= 0 && y < m_height))
                    {
                        int nindex = y * m_width + x;
                        if (newlabels[nindex] >= 0) {
                            adjlabel = newlabels[nindex];
                        }
                    }
                }}

                int count = 1;
                {for (int c = 0; c < count; c++)
                {
                    for (int n = 0; n < 4; n++)
                    {
                        int x = xvec[c] + dx[n];
                        int y = yvec[c] + dy[n];
                        if ((x >= 0 && x < m_width) && (y >= 0 && y < m_height))
                        {
                            int nindex = y * m_width + x;
                            if (newlabels[nindex] < 0 &&
                                oldlabels[oindex] == oldlabels[nindex])
                            {
                                xvec[count] = x;
                                yvec[count] = y;
                                newlabels[nindex] = label;
                                count++;
                            }
                        }
                    }
                }}

                if (count <= spixSz >> 2)
                {
                    for (int c = 0; c < count; c++)
                    {
                        int ind = yvec[c] * m_width + xvec[c];
                        newlabels[ind] = adjlabel;
                    }
                    label--;
                }
                label++;
            }
            oindex++;
        }
    }
    //...
}

void SlicAlgorithm::constructSpxMat(cv::Mat & labeled)
{
    if (!spxmat.empty()) spxmat.clear();
    int* rowptr = nullptr;
    for (int y = 0; y < labeled.rows; y++)
    {
        rowptr = labeled.ptr<int>(y);
        for (int x = 0; x < labeled.cols; x++)
            spxmat[rowptr[x]].push_back(cv::Point2i(x, y));
    }
}

void SlicAlgorithm::constructSpxMat(
    std::vector<int> & labeled,
    const int & width,
    const int & height)
{
    if (!spxmat.empty()) spxmat.clear();

    int idx = 0;
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
            spxmat[labeled[idx++]].push_back(cv::Point2i(x, y));
}

void SlicAlgorithm::colorSegments(cv::Mat & labeled)
{
    cv::Mat colored = cv::Mat::zeros(labeled.size(), CV_32SC3);

    if (spxmat.empty())
        constructSpxMat(labeled);

    cv::RNG rng(12345);

    for (auto & spx : spxmat)
    {
        // auto sc = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
        auto sc = rng.uniform(0, 255);
        for (auto & pnt : spx.second)
        {
            auto & pixel = colored.at<cv::Vec3i>(pnt);
            pixel[0] = sc;
            pixel[0] = sc;
            pixel[0] = sc;
        }
    }
    labeled = colored;
}

std::vector<cv::Point2i> & SlicAlgorithm::getSpixCoords(int spxid)
{
    return spxmat[spxid];
}

size_t SlicAlgorithm::getSpixSize(int spxid)
{
    return spxmat[spxid].size();
}

void SlicAlgorithm::findSpixNeighbors(
    nmap & neighbors,
    cv::Mat &        labeled)
{
    if (!neighbors.empty())
        neighbors.clear();

    int* rowptr0 = nullptr, *rowptr1 = nullptr;
    rowptr0 = labeled.ptr<int>(0);
    for (int x = 1; x < labeled.cols - 1; x++) {
        neighbors[rowptr0[x]].insert(rowptr0[x - 1]);
        neighbors[rowptr0[x]].insert(rowptr0[x + 1]);
    }
    for (int y = 1; y < labeled.rows; y++)
    {
        rowptr0 = labeled.ptr<int>(y - 1);
        rowptr1 = labeled.ptr<int>(y);
        for (int x = 1; x < labeled.cols - 1; x++)
        {
            neighbors[rowptr1[x]].insert(rowptr0[x]);
            neighbors[rowptr1[x]].insert(rowptr1[x - 1]);
            neighbors[rowptr1[x]].insert(rowptr1[x + 1]);
        }
    }
}