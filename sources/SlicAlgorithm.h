#ifndef SLIC_ALGO_h
#define SLIC_ALGO_h

#include <opencv2/opencv.hpp>
#include <unordered_map>
#include <unordered_set>

class SlicAlgorithm
{
public:
    using nmap = std::unordered_map<int, std::unordered_set<int>>;

    SlicAlgorithm();
    virtual ~SlicAlgorithm() = default;

    void makeSegmentation(
        cv::Mat &          img,
        cv::Mat &          labeled,
        const int &        width,
        const int &        height,
        std::vector<int> & klabels,
        std::vector<int> & labelids,
        const int &        spixsize,
        const double &     compactness,
        bool               enforce = true
    );

    static void makeContoursAroundSpixs(
        cv::Mat &            img,
        std::vector<int> &   labels,
        const int &          width,
        const int &          height,
        const unsigned int & color
    );

    void makeContours(
        cv::Mat &          img,
        std::vector<int> & labels,
        const int &        width,
        const int &        height
    );

    void constructSpxMat(
        cv::Mat & labeled
    );

    void constructSpxMat(
        std::vector<int> & labeled,
        const int & width,
        const int & height
    );

    void colorSegments(
        cv::Mat & labeled
    );

    int setSpxToValue(
        int       spxid,
        cv::Mat & mtx,
        int       value)
    {
        int numOfProcessedSpix = 0;
        auto first = spxmat[spxid].begin();
        auto last = spxmat[spxid].end();
        cv::Vec3i* rowptr = nullptr;

        while (first != last)
        {
            auto prev = first++;
            while (first != last && prev != last && prev->y == first->y)
                first++;
            rowptr = mtx.ptr<cv::Vec3i>(prev->y);
            while (prev != first)
            {
                auto & pixel = rowptr[prev->x];
                pixel[0] = pixel[1] = pixel[2] = value;
                prev++;
                numOfProcessedSpix++;
            }
        }
        return numOfProcessedSpix;
    }
    
    template <typename T>
    int setSpxToValue(
        int              spxid,
        std::vector<T> & mtx,
        int              width,
        T                value)
    {
        int numOfProcessedSpix = 0;
        auto first = spxmat[spxid].begin();
        auto last = spxmat[spxid].end();
        T* rowptr = nullptr;

        while (first != last)
        {
            auto prev = first++;
            while (first != last && prev != last && prev->y == first->y)
                first++;
            rowptr = &mtx[(prev->y) * width];
            while (prev != first)
            {
                rowptr[prev->x] = value;
                prev++;
                numOfProcessedSpix++;
            }
        }
        return numOfProcessedSpix;
    }

    std::vector<cv::Point2i> & getSpixCoords(
        int spxid
    );

    size_t getSpixSize(
        int spxid
    );

    void findSpixNeighbors(
        nmap & neighbors,
        cv::Mat &        labeled
    );

private:
    struct Seed
    {
        double L;
        double A;
        double B;
        double x;
        double y;

        void reset() { L = A = B = x = y = 0.0; }
    };

    void performSegmentation(
        std::vector<Seed> & kseeds,
        std::vector<int> &  klabels,
        const int &         step,
        const double &      m = 10.0
    );

    void initSeedsOfSpixs(
        std::vector<Seed> & kseeds,
        const int &         step
    );

    void enforceLabelConnectivity(
        std::vector<int> & klabeled,
        std::vector<int> & labelids,
        const int &        spixsize
    );

    void enforce(
        std::vector<int> & klabeled,
        std::vector<int> & newlabels,
        const int & K
    );

private:
    int     m_width;
    int     m_height;
    cv::Mat labimg;

    std::unordered_map<int, std::vector<cv::Point2i>> spxmat;
};

#endif