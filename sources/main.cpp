#include <iostream>
#include <unordered_set>
#include <chrono>

#include "SlicAlgorithm.h"
#include "Histogram.h"
#include "StereoAlgorithm.h"
#include "WeightedGraph.h"
#include "StereoAlgorithm.h"
#include "ArgsParser.h"
#include "Tester.h"

void TestHistogram()
{
    int numOfBins = 20;
    float low = 0.7f, high = 1.1f;
    cv::Mat mtx(10, 2, CV_32FC1);
    cv::randu(mtx, cv::Scalar(low), cv::Scalar(2.2));

    std::cout << "Mtx before hist calculation : " << mtx << "\n";

    std::vector<float> refs;
    refs.reserve(mtx.rows * mtx.cols);

    std::vector<int> counts;
    counts.reserve(numOfBins);

    float* rowptr = nullptr;

    for (int y = 0; y < mtx.rows; y++)
    {
        rowptr = mtx.ptr<float>(y);
        for (int x = 0; x < mtx.cols; x++)
            refs.push_back(rowptr[x]);
    }

    Histogram<float> hist(numOfBins, low, high);
    counts.assign(numOfBins, 0);
    hist.calculateHist(refs, counts);

    std::cout << "Samples after sorting :\n";
    for (const auto & c : counts)
        std::cout << c << " ";
    std::cout << "\n";

    std::cout << "Mtx after hist calculation : " << mtx << "\n";
}

void TestMaxTripleSum()
{
    std::vector<int> data;
    data.reserve(5);
    data.push_back(1);
    data.push_back(9);
    data.push_back(10);
    data.push_back(1);
    data.push_back(5);

    std::cout << "Max sum : " << SlicBasedStereoM::maxTripleSum(data) << "\n";
}

void TestUnorderedSet()
{
    std::unordered_set<int> unique_values;
    unique_values.insert(1);
    unique_values.insert(1);
    unique_values.insert(1);
    unique_values.insert(2);
    unique_values.insert(2);
    unique_values.insert(2);
    unique_values.insert(2);
    unique_values.insert(3);
    unique_values.insert(3);
    unique_values.insert(4);
    unique_values.insert(4);
    unique_values.insert(4);

    for (const auto & el : unique_values)
        std::cout << el << " ";
    std::cout << "\n";
}

void TestDivision0()
{
    int W = 5, H = 5;
    cv::Mat mtx0 = cv::Mat::zeros(H, W, CV_32FC1);
    cv::Mat mtx1 = cv::Mat::zeros(H, W, CV_32FC1);
    float* rowptr0 = nullptr, *rowptr1 = nullptr;

    for (int y = 0; y < H; y++)
    {
        rowptr0 = mtx0.ptr<float>(y);
        rowptr1 = mtx1.ptr<float>(y);
        for (int x = 0; x < W; x++) {
            rowptr0[x] = y * W + x;
            rowptr1[x] = rowptr0[x];
        }
    }

    std::cout << "mtx0 :\n" << mtx0 << "\n";
    std::cout << "mtx1 :\n" << mtx1 << "\n";

    cv::Mat div0 = mtx0 / mtx1;
    std::cout << "div0 :\n" << div0 << "\n";

    // window = im(Range(ymin, ymax), Range(xmin, xmax));
    cv::Mat div1 = mtx0(cv::Range(0, 4), cv::Range(2, 4)) / mtx1(cv::Range(0, 4), cv::Range(0, 2));
    std::cout << "div1 :\n" << div1 << "\n";
}

void DumbDivision(cv::Mat & mtx0, cv::Mat & mtx1, cv::Mat & div, int disp)
{
    int m_height = mtx0.rows, m_width = mtx0.cols;
    float *limgptr = nullptr, *rimgptr = nullptr, *ratioptr = nullptr;

    div = cv::Mat::zeros(m_height, m_width, mtx0.type());

    //int offset = 0;

    for (int d = 0; d < disp; d++)
    {
        for (int y = 0; y < m_height; y++)
        {
            limgptr  = mtx0.ptr<float>(y);
            rimgptr  = mtx1.ptr<float>(y);
            ratioptr = div.ptr<float>(y);

            for (int x = 0; x < m_width; x++)
            {
                if (x + d < m_width)
                    ratioptr[x] = limgptr[x] / rimgptr[x + d];
            }
        }
    }
}

void TestDivision1()
{
    int W = 1000, H = 1000;
    cv::Mat mtx0 = cv::Mat::zeros(H, W, CV_32FC1);
    cv::Mat mtx1 = cv::Mat::zeros(H, W, CV_32FC1);
    float* rowptr0 = nullptr, *rowptr1 = nullptr;

    cv::randu(mtx0, cv::Scalar(0.0f), cv::Scalar(2.2f));
    cv::randu(mtx1, cv::Scalar(0.0f), cv::Scalar(1.1f));

    //std::cout << "mtx0 :\n" << mtx0 << "\n";
    //std::cout << "mtx1 :\n" << mtx1 << "\n";

    {auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; i++) {
        //cv::Mat div00 = mtx0 / mtx1;
        for (int d = 0; d < 73; d++) {
            cv::Mat div1 = mtx0(cv::Range(0, H), cv::Range(d, W)) / mtx1(cv::Range(0, H), cv::Range(0 + d, W));
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    double timeTaken = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    timeTaken *= 1e-9;
    std::cout << "div00 time (opencv) :\n" << timeTaken << "\n";}

    {auto start = std::chrono::high_resolution_clock::now();
    cv::Mat div01;
    for (int i = 0; i < 100; i++)
        DumbDivision(mtx0, mtx1, div01, 73);
    auto end = std::chrono::high_resolution_clock::now();
    double timeTaken = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    timeTaken *= 1e-9;
    std::cout << "div01 time (opencv) :\n" << timeTaken << "\n"; }

    // window = im(Range(ymin, ymax), Range(xmin, xmax));
    //cv::Mat div1 = mtx0(cv::Range(0, 4), cv::Range(2, 4)) / mtx1(cv::Range(0, 4), cv::Range(0, 2));
    //std::cout << "div1 :\n" << div1 << "\n";
}

void TestSlic(int argc, const char** argv)
{
    // Input Variables
    std::string input, output;
    int nx, ny;
    int m;

    // Default values
    output = "out.png";
    nx = 15;
    ny = 15;
    m = 20;

    if (argc == 1 || argc == 4) {
        std::cout << "SLIC superpixel segmentation" << std::endl;
        std::cout << "Usage: SLIC INPUT OUTPUT [nx ny] [m]" << std::endl;
        return;
    }
    if (argc >= 2)
        input = argv[1];
    if (argc >= 3)
        output = argv[2];
    if (argc >= 5) {
        nx = atoi(argv[3]);
        ny = atoi(argv[4]);
    }
    if (argc == 6)
        m = atoi(argv[5]);

    // Read in image
    cv::Mat im = cv::imread(input);

    if (!im.data) {
        std::cerr << "no image data at " << input << std::endl;
        return;
    }

    cv::Mat labeled;
    SlicAlgorithm algo;
    std::vector<int> klabels, labelIDs;
    cv::Mat blurred;
    if (im.type() != CV_32F)
        im.convertTo(im, CV_32F, 1.0 / 255.0);
    cv::GaussianBlur(im, blurred, cv::Size(3, 3), 0, 0);
    algo.makeSegmentation(blurred, labeled, im.cols, im.rows, klabels, labelIDs, nx*ny, m);

    //SlicAlgorithm::nmap nmap;
    //algo.findSpixNeighbors(nmap, labeled);

    //algo.colorSegments(labeled);
    //algo.setSpxToValue(0, labeled, 0);
    
    // testing neighbors validity
    //for (const auto & n : nmap[25])
    //{
    //    if (n != 25)
    //        algo.setSpxToValue(n, im, 0);
    //}

    //algo.setSpxToValue(0, im, 0);
    
    algo.makeContoursAroundSpixs(im, klabels, im.cols, im.rows, 0);
    cv::imwrite(output + "_COLORED.png", labeled);
    cv::imwrite(output + "SPX0.png", 255*im);
}

void TestGraph()
{
    SlicAlgorithm::nmap nmap;
    nmap[0].insert(1); nmap[0].insert(2); nmap[0].insert(0);
    nmap[1].insert(0); nmap[1].insert(2); nmap[1].insert(4); nmap[1].insert(3); nmap[1].insert(1);
    nmap[2].insert(0); nmap[2].insert(1); nmap[2].insert(4); nmap[2].insert(5); nmap[2].insert(6); nmap[2].insert(2);
    nmap[3].insert(1); nmap[3].insert(4); nmap[3].insert(7); nmap[3].insert(3);
    nmap[4].insert(1); nmap[4].insert(2); nmap[4].insert(5); nmap[4].insert(8); nmap[4].insert(7); nmap[4].insert(3); nmap[4].insert(4);
    nmap[5].insert(6); nmap[5].insert(2); nmap[5].insert(4); nmap[5].insert(8); nmap[5].insert(5);
    nmap[6].insert(2); nmap[6].insert(5); nmap[6].insert(6);
    nmap[7].insert(3); nmap[7].insert(4); nmap[7].insert(8); nmap[7].insert(7);
    nmap[8].insert(5); nmap[8].insert(4); nmap[8].insert(7); nmap[8].insert(8);
    
    WeightedGraph graph;
    graph.construct(nmap);
    std::cout << "G :\n" << graph << "\n";
    int counter = 1;
    graph.traverse([&counter](int, int) -> double {
        return counter++;
    });
    std::cout << "\nG :\n" << graph << "\n";
}

void Run(int argc, const char** argv)
{
    Properties props;
    ArgumentsParser argpsr(argc, argv);
    argpsr.showProperties();

    if (!argpsr.tryGetProperties(props)) {
        std::cout << argpsr.errorMessage() << "\n";
        return;
    }
    
    std::cout << "\nObtained input values : \n" << props << "\n";

    // Read in image
    cv::Mat im0 = cv::imread(props.inputDir + "im0.png");
    cv::Mat im1 = cv::imread(props.inputDir + "im1.png");

    if (!im0.data || !im1.data) {
        std::cerr << "no image data at " << props.inputDir << "\n";
    }

    SlicBasedStereoM algo(props);
    cv::Mat disparityMap;//, zMap;
    algo.match(im0, im1, disparityMap);
    algo.getSlic().makeContoursAroundSpixs(im0, algo.getKLabels(), im0.cols, im0.rows, 0);
    //algo.makeZMap(zMap, disparityMap);
    cv::imwrite(props.outputDir + "myDisp.png",    disparityMap);
    cv::imwrite(props.outputDir + "myLabels.png",  im0*255);
    //cv::imwrite(props.outputDir + "myZMap0.png",   zMap*255);
    //cv::imwrite(props.outputDir + "myZMap1.png",   zMap);
    //utils::save_pfm_file(props.outputDir + "zmap.pfm", zMap);
    
    //cv::Mat cnn;
    //cv::bilateralFilter(disparityMap, cnn, 9, 75, 75, cv::BORDER_DEFAULT);
    //cv::GaussianBlur(disparityMap, cnn, cv::Size(9, 9), 0, 0, cv::BORDER_DEFAULT);
    disparityMap.convertTo(disparityMap, CV_32FC1);
    //cnn.convertTo(cnn, CV_32FC1);
    utils::save_pfm_file(props.outputDir + "disp.pfm", disparityMap);
    //utils::save_pfm_file(props.outputDir + "cnn.pfm", cnn);
}

void Test(int argc, const char** argv)
{
    Properties props;
    ArgumentsParser argpsr(argc, argv);
    argpsr.showProperties();

    if (!argpsr.tryGetProperties(props)) {
        std::cout << argpsr.errorMessage() << "\n";
        return;
    }

    std::cout << "\nObtained input values : \n" << props << "\n";

    // Read in image
    cv::Mat im0 = cv::imread(props.inputDir + "im0.png");
    cv::Mat im1 = cv::imread(props.inputDir + "im1.png");

    if (!im0.data || !im1.data) {
        std::cerr << "no image data at " << props.inputDir << "\n";
    }

    Tester tester(props.inputDir, props.outputDir, props.calib);
    SlicBasedStereoM algo0(props);
    tester.test(im0, im1, algo0, "_SLICBM", 2.0);
    algo0.getSlic().makeContoursAroundSpixs(im0, algo0.getKLabels(), im0.cols, im0.rows, 0);
    //algo.makeZMap(zMap, disparityMap);
    cv::imwrite(props.outputDir + "myLabels.png", im0 * 255);

    SemiGlobalBM algo1(props);
    tester.test(im0, im1, algo1, "_SGBM", 2.0);
}

int main(int argc, const char** argv)
{
    //std::unordered_map<int, std::vector<int>> map;
    //for (int i = 0; i < 10; i++)
    //    map[i].push_back(i);
    //for (int i = 0; i < 10; i++)
    //    map[i].push_back(i);
    //for (auto & item : map)
    //{
    //    std::cout << item.first << " : ";
    //    for (auto & i : item.second)
    //        std::cout << i << " ";
    //    std::cout << "\n";
    //}
    
    //----------------------------------------------------------------------------------------
    //TestHistogram();
    //TestMaxTripleSum();
    //TestUnorderedSet();
    //TestSlic(argc, argv);
    //TestGraph();
    //TestDivision0();
    //TestDivision1();
    //----------------------------------------------------------------------------------------

    {// Input Variables
    //std::string input, output;
    //int nx = 40, ny = 40, m = 20;

    //if (argc == 1 || argc == 4) {
    //    std::cout << "SLIC superpixel segmentation" << std::endl;
    //    std::cout << "Usage: SLIC INPUT_DIR OUTPUT_DIR NX NY M" << std::endl;
    //    return -1;
    //}
    //if (argc >= 2)
    //    input = argv[1];
    //if (argc >= 3)
    //    output = argv[2];
    //if (argc >= 5) {
    //    nx = atoi(argv[3]);
    //    ny = atoi(argv[4]);
    //}
    //if (argc == 6)
    //    m = atoi(argv[5]);

    //// Read in image
    //cv::Mat im0 = cv::imread(input + "\\im0.png");
    //cv::Mat im1 = cv::imread(input + "\\im1.png");

    //if (!im0.data || !im1.data) {
    //    std::cerr << "no image data at " << input << "\n";
    //    return -1;
    //}

    //Properties props;
    //props.calib.read(input + "\\calib.txt");
    //props.superpixsz = nx * ny;
    //props.compactness = m;

    //cv::Mat disparityMap;
    //SlicBasedStereoM algo(props);
    //algo.setFileName(output + "\\COSTS.xml");
    //algo.match(im0, im1, disparityMap);

    //cv::imwrite(output + "\\myDisp.png", disparityMap);

    //cv::FileStorage file(output + "\\LABELED.xml", cv::FileStorage::WRITE);
    //file << "LABELED_MTX" << disparityMap;

    //std::vector<int> data{ 0, 1, 2, 3, 4, 5, 6 };
    //file << "Vector" << data;
    }

    Run(argc, argv);
    //TestSlic(argc, argv);
    
    //Test(argc, argv);
    system("pause");
    return 0;
}