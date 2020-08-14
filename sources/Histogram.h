#ifndef HISTOGRAM_H
#define HISTOGRAM_H

#include <vector>
#include <algorithm>
#include <fstream>

template <typename T>
class Histogram
{
public:
    Histogram(int bins, T low, T high)
        : m_numOfBins{bins}, m_low{low}, m_high{high}
    {}

    virtual ~Histogram() = default;

    void calculateHist(
        std::vector<T> &   sample,
        std::vector<int> & hist,
        std::string fileName = "")
    {
        if (m_low >= m_high || m_numOfBins < 2)
            return;

        std::sort(sample.begin(), sample.end()/*, [](const T & a, const T & b) -> bool { return a <= b; }*/);

        const T delta = (m_high - m_low) / static_cast<T>(m_numOfBins);

        // generating ranges
        m_ranges.reserve(m_numOfBins + 1);
        m_ranges.push_back(m_low);
        for (int idx = 0; idx < m_numOfBins; idx++)
            m_ranges.push_back(m_ranges[idx] + delta);

        // finding subset of values from sample, which falls into ( leftBound; rightBound ]
        auto first = sample.begin();
        while (first != sample.end() && *first < m_low) first++;
        
        auto last = first;
        if (last != sample.end()) last++;
        while (last != sample.end() && *last <= m_high) last++;

        // now [ first; last ] contains appropriate values
        int b = 0;
        while (first != last && b < m_numOfBins)
        {
            if (m_ranges[b] < *first && *first <= m_ranges[b + 1])
            {
                hist[b]++;
                first++;
            }
            else b++;
        }
        if (last != sample.end() && m_ranges[b] < *last && *last <= m_ranges[b + 1])
            hist[b - 1]++;
    }

private:
    int m_numOfBins;
    T   m_low, m_high;
    std::vector<T>   m_ranges;
    std::vector<int> m_markers;
};

#endif