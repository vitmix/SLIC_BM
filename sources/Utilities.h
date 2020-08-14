/********************************************************************
 * This code is copied from taniai
 * Interface functions to read and write PFM files and cost volumes
 *******************************************************************/

#ifndef UTILITIES_H
#define UTILITIES_H

#include <opencv2/opencv.hpp>
#include <string>

namespace utils
{
    template <typename T>
    T ConvertToValue(const std::string & data)
    {
        return static_cast<T>(std::stod(data));
    }

    template <>
    float ConvertToValue(const std::string & data)
    {
        return std::stof(data);
    }

    template <>
    int ConvertToValue(const std::string & data)
    {
        return std::stoi(data);
    }

    template <>
    std::string ConvertToValue(const std::string & data)
    {
        return data;
    }

    template <>
    bool ConvertToValue(const std::string & data)
    {
        return data == "true" ? true : data == "false" ? false : ConvertToValue<int>(data) != 0;
    }

    inline bool Contains(const std::string & str1, const std::string & str2)
    {
        std::string::size_type pos = str1.find(str2);

        if (pos == std::string::npos)
            return false;

        return true;
    }

    using byte = uchar;

    static int is_little_endian()
    {
        if (sizeof(float) != 4)
        {
            printf("Bad float size.\n");
            exit(1);
        }
        byte b[4] = { 255, 0, 0, 0 };
        return *((float *)b) < 1.0;
    }

    static cv::Mat read_pfm_file(const std::string & filename)
    {
        int w, h;
        char buf[256];
        FILE *f = fopen(filename.c_str(), "rb");
        if (f == NULL)
        {
            //wprintf(L"PFM file absent: %s\n\n", filename.c_str());
            return cv::Mat();
        }

        int channel = 1;
        fscanf(f, "%s\n", buf);
        if (strcmp(buf, "Pf") == 0) channel = 1;
        else if (strcmp(buf, "PF") == 0) channel = 3;
        else {
            printf(buf);
            printf("Not a 1/3 channel PFM file.\n");
            return cv::Mat();
        }
        fscanf(f, "%d %d\n", &w, &h);
        double scale = 1.0;
        fscanf(f, "%lf\n", &scale);
        int little_endian = 0;
        if (scale < 0.0)
        {
            little_endian = 1;
            scale = -scale;
        }
        size_t datasize = w * h*channel;
        std::vector<byte> data(datasize * sizeof(float));

        cv::Mat image = cv::Mat(h, w, CV_MAKE_TYPE(CV_32F, channel));

        // Adjust the position of the file because fscanf() reads too much (due to "\n"?)
        fseek(f, -(long)datasize * sizeof(float), SEEK_END);
        size_t count = fread((void *)&data[0], sizeof(float), datasize, f);
        if (count != datasize)
        {
            printf("Expected size %d, read size %d.\n", datasize, count);
            printf("Could not read ground truth file.\n");
            return cv::Mat();
        }
        int native_little_endian = is_little_endian();
        for (size_t i = 0; i < datasize; i++) {
            byte* p = &data[i * 4];
            if (little_endian != native_little_endian) {
                byte temp;
                temp = p[0];
                p[0] = p[3];
                p[3] = temp;
                temp = p[1];
                p[1] = p[2];
                p[2] = temp;
            }
            int jj = (i / channel) % w;
            int ii = (i / channel) / w;
            int ch = i % channel;
            image.at<float>(h - 1 - ii, jj * channel + ch) =
                *((float*)p);
        }
        fclose(f);
        return image;
    }

    static void save_pfm_file(const std::string & filename, const cv::Mat & image)
    {
        int width = image.cols;
        int height = image.rows;

        FILE *stream = fopen(filename.c_str(), "wb");
        if (stream == NULL)
        {
            //wprintf(L"PFM file absent: %s\n\n", filename.c_str());
            return;
        }
        // write the header: 3 lines: Pf, dimensions, scale factor (negative val == little endian)
        int channel = image.channels();
        if (channel == 1)
            fprintf(stream, "Pf\n%d %d\n%lf\n", width, height, -1.0 / 255.0);
        else if (channel == 3)
            fprintf(stream, "PF\n%d %d\n%lf\n", width, height, -1.0 / 255.0);
        else {
            printf("Channels %d must be 1 or 3\n", image.channels());
            return;
        }

        // pfm stores rows in inverse order!
        int linesize = width * channel;
        std::vector<float> rowBuff(linesize);
        for (int y = height - 1; y >= 0; y--)
        {
            auto ptr = image.ptr<float>(y);
            auto pBuf = &rowBuff[0];
            for (int x = 0; x < linesize; x++)
            {
                float val = (float)(*ptr);
                pBuf[x] = val;
                ptr++;
                /*if (val > 0 && val <= 255)
                rowBuf[x] = val;
                else
                {
                printf("invalid: val %f\n", flo(x,y));
                rowBuf[x] = 0.0f;
                }*/
            }
            if ((int)fwrite(&rowBuff[0], sizeof(float), width, stream) != width)
            {
                printf("[ERROR] problem with fwrite.");
            }
            fflush(stream);
        }

        fclose(stream);
        return;
    }

    enum class Direction
    {
        ROW = 0, // X-axis
        COLUMN // Y-axis
    };

    cv::Mat CumulativeSum(const cv::Mat & in, Direction direct)
    {
        int H = in.rows, W = in.cols;
        cv::Mat out = cv::Mat::zeros(H, W, in.type());

        double* outPtr = nullptr;
        const double* inPtr = nullptr;

        if (direct == Direction::ROW)
        {
            for (int y = 0; y < H; y++)
            {
                outPtr = static_cast<double*>(out.ptr<double>(y));
                inPtr = static_cast<const double*>(in.ptr<double>(y));
                outPtr[0] = inPtr[0];

                for (int x = 1; x < W; x++)
                    outPtr[x] = outPtr[x - 1] + inPtr[x];
            }
        }
        else // Direction::COLUMN
        {
            double* prevPtr = nullptr;
            outPtr = static_cast<double*>(out.ptr<double>(0));
            inPtr = static_cast<const double*>(in.ptr<double>(0));

            for (int x = 0; x < W; x++)
                outPtr[x] = inPtr[x];

            for (int y = 1; y < H; y++)
            {
                outPtr = static_cast<double*>(out.ptr<double>(y));
                inPtr = static_cast<const double*>(in.ptr<double>(y));
                prevPtr = static_cast<double*>(out.ptr<double>(y - 1));

                for (int x = 0; x < W; x++)
                    outPtr[x] = prevPtr[x] + inPtr[x];
            }
        }

        return out;
    }

    cv::Mat BoxFilterSummation(const cv::Mat & in, int rad)
    {
        int H = in.rows, W = in.cols;
        CV_Assert(W >= rad && H >= rad);

        cv::Mat out = cv::Mat::zeros(H, W, in.type());
        cv::Mat cumSum = CumulativeSum(in, Direction::COLUMN);

        double *outPtr = nullptr, *prevPtr = nullptr, *nextPtr = nullptr;

        for (int y = 0; y < rad + 1; y++)
        {
            outPtr = static_cast<double*>(out.ptr<double>(y));
            nextPtr = static_cast<double*>(cumSum.ptr<double>(y + rad));

            for (int x = 0; x < W; x++)
                outPtr[x] = nextPtr[x];
        }

        for (int y = rad + 1; y < H - rad; y++)
        {
            outPtr = static_cast<double*>(out.ptr<double>(y));
            prevPtr = static_cast<double*>(cumSum.ptr<double>(y - rad - 1));
            nextPtr = static_cast<double*>(cumSum.ptr<double>(y + rad));

            for (int x = 0; x < W; x++)
                outPtr[x] = nextPtr[x] - prevPtr[x];
        }

        for (int y = H - rad; y < H; y++)
        {
            outPtr = static_cast<double*>(out.ptr<double>(y));
            prevPtr = static_cast<double*>(cumSum.ptr<double>(y - rad - 1));
            nextPtr = static_cast<double*>(cumSum.ptr<double>(H - 1));

            for (int x = 0; x < W; x++)
                outPtr[x] = nextPtr[x] - prevPtr[x];
        }

        cumSum = CumulativeSum(out, Direction::ROW);

        for (int y = 0; y < H; y++)
        {
            outPtr = static_cast<double*>(out.ptr<double>(y));
            nextPtr = static_cast<double*>(cumSum.ptr<double>(y));

            for (int x = 0; x < rad + 1; x++)
                outPtr[x] = nextPtr[x + rad];

            for (int x = rad + 1; x < W - rad; x++)
                outPtr[x] = nextPtr[x + rad] - nextPtr[x - rad - 1];

            for (int x = W - rad; x < W; x++)
                outPtr[x] = nextPtr[W - 1] - nextPtr[x - rad - 1];
        }

        return out;
    }
}

#endif // !UTILITIES_H