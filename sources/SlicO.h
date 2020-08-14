#ifndef SLICO_h
#define SLICO_h

#include <opencv2/opencv.hpp>

class SlicO
{
private:
    struct Seed
    {
        double L;
        double A;
        double B;
        double x;
        double y;

        void set(double val) { L = A = B = x = y = val; }
    };

public:
    SlicO() : m_width{0}, m_height{0} {}
    virtual ~SlicO() = default;

    void makeSegmentation(
        cv::Mat & img,
        cv::Mat & labeled,
        std::vector<int> & klabels,
        std::vector<int> & labelids,
        const int & step)
    {
        m_width = img.cols; m_height = img.rows;
        int sz = m_width * m_height;
        
        std::vector<Seed> kseeds;
        std::vector<int> initLabels(sz, -1);
        std::vector<double> edgeMag(0);

        if (img.type() != CV_32F)
            img.convertTo(img, CV_32F, 1.0 / 255.0);
        cv::cvtColor(img, m_lab, cv::COLOR_BGR2Lab);


    }

private:
    void detectLabEdges(std::vector<double> & edges)
    {
        int sz = m_width * m_height;
        edges.resize(sz, 0.0);

        cv::Vec3f *prevptr, *currptr, *nextptr;
        prevptr = currptr = nextptr = nullptr;

        for (int y = 1; y < (m_height - 1); y++)
        {
            prevptr = m_lab.ptr<cv::Vec3f>(y - 1);
            currptr = m_lab.ptr<cv::Vec3f>(y + 0);
            nextptr = m_lab.ptr<cv::Vec3f>(y + 1);

            int idx = y * m_width;

            for (int x = 1; x < (m_width - 1); x++)
            {
                cv::Vec3f & pntDx0 = currptr[x - 1];
                cv::Vec3f & pntDx1 = currptr[x + 1];
                
                double dx = (pntDx0[0] - pntDx1[0]) * (pntDx0[0] - pntDx1[0]) +
                            (pntDx0[1] - pntDx1[1]) * (pntDx0[1] - pntDx1[1]) +
                            (pntDx0[2] - pntDx1[2]) * (pntDx0[2] - pntDx1[2]);

                cv::Vec3f & pntDy0 = prevptr[x];
                cv::Vec3f & pntDy1 = nextptr[x];

                double dy = (pntDy0[0] - pntDy1[0]) * (pntDy0[0] - pntDy1[0]) +
                            (pntDy0[1] - pntDy1[1]) * (pntDy0[1] - pntDy1[1]) + 
                            (pntDy0[2] - pntDy1[2]) * (pntDy0[2] - pntDy1[2]);

                edges[idx + x] = std::sqrt(dx) + std::sqrt(dy);
            }
        }
    }

    void initSeeds(
        std::vector<Seed> & kseeds,
        std::vector<double> & edgemag,
        const int & step)
    {
        int numSeeds = 0, n = 0;
        
        int xstrips = (0.5 + static_cast<double>(m_width) / static_cast<double>(step));
        int ystrips = (0.5 + static_cast<double>(m_height) / static_cast<double>(step));

        int xerr = m_width - step * xstrips;
        int yerr = m_height - step * ystrips;

        if (xerr < 0) { xstrips--; xerr = m_width - xstrips * step; }
        if (yerr < 0) { ystrips--; yerr = m_height - ystrips * step; }

        double xerrperstrip = static_cast<double>(xerr) / static_cast<double>(xstrips);
        double yerrperstrip = static_cast<double>(yerr) / static_cast<double>(ystrips);

        int xoff = step / 2, yoff = step / 2, yseed = 0, xseed = 0;
        cv::Vec3f *rowptr = nullptr;

        numSeeds = xstrips * ystrips;
        kseeds.resize(numSeeds);

        for (int y = 0; y < ystrips; y++)
        {
            yseed = (y * step) + yoff + (y * yerrperstrip);
            rowptr = m_lab.ptr<cv::Vec3f>(yseed);

            for (int x = 0; x < xstrips; x++)
            {
                xseed = x * step + xoff + (x * xerrperstrip);
                cv::Vec3f & pnt = rowptr[xseed];

                Seed & seed = kseeds[n];
                seed.L = pnt[0];
                seed.A = pnt[1];
                seed.B = pnt[2];
                seed.x = xseed;
                seed.y = yseed;
                n++;
            }
        }
    
        // seed perturbation
        perturbSeeds(kseeds, edgemag);
    }

    void perturbSeeds(
        std::vector<Seed> & kseeds,
        std::vector<double> & edgemag)
    {
        const int dx[8] = { -1, -1,  0,  1, 1, 1, 0, -1 };
        const int dy[8] = { 0, -1, -1, -1, 0, 1, 1,  1 };

        int numSeeds = static_cast<int>(kseeds.size());

        for (int n = 0; n < numSeeds; n++)
        {
            int ox = kseeds[n].x;
            int oy = kseeds[n].y;
            int oind = oy * m_width + ox;
            int storeind = oind;

            for (int i = 0; i < 8; i++)
            {
                int nx = ox + dx[i];
                int ny = oy + dy[i];

                if (nx >= 0 && nx < m_width && ny >= 0 && ny < m_height)
                {
                    int nind = ny * m_width + nx;
                    if (edgemag[nind] < edgemag[storeind]) {
                        storeind = nind;
                    }
                }
            }

            if (storeind != oind)
            {
                Seed & seed = kseeds[n];
                seed.x = storeind % m_width;
                seed.y = storeind / m_width;

                cv::Vec3f & pnt = m_lab.at<cv::Vec3f>(seed.y, seed.x);
                seed.L = pnt[0];
                seed.A = pnt[1];
                seed.B = pnt[2];
            }
        }
    }

private:
    int     m_width;
    int     m_height;
    cv::Mat m_lab;
};

#endif