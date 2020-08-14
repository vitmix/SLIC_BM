#ifndef PROPERTIES_H
#define PROPERTIES_H

#include <string>

struct CalibrationParams
{
    float cam0[3][3], cam1[3][3];
    float doffs = 0.0;
    float baseline = 0.0;
    int   imWidth = 0, imHeight = 0;
    int   ndisp = 0;
    int   isint = 0;
    int   vmin = 0, vmax = 0;
    float dyavg = 0.0, dymax = 0.0;

    bool read(const std::string & filename)
    {
        auto FileCloser = [](FILE * fptr) {
            if (fptr != nullptr)
                fclose(fptr);
        };

        auto ShowErrorMsg = [](const std::string & param) {
            std::cout << "Error while parsing " + param + " was encountered...\n";
        };

        std::unique_ptr<FILE, decltype(FileCloser)> FileHandler(fopen(filename.c_str(), "r"), FileCloser);
        char buffer[512];

        if (FileHandler != nullptr)
        {
            if (fgets(buffer, sizeof(buffer), FileHandler.get()) == nullptr ||
                sscanf(buffer, "cam0 = [%f %f %f; %f %f %f; %f %f %f]\n",
                    &cam0[0][0], &cam0[0][1], &cam0[0][2],
                    &cam0[1][0], &cam0[1][1], &cam0[1][2],
                    &cam0[2][0], &cam0[2][1], &cam0[2][2]) != 9)
            {
                ShowErrorMsg("cam0"); return false;
            }

            if (fgets(buffer, sizeof(buffer), FileHandler.get()) == nullptr ||
                sscanf(buffer, "cam1 = [%f %f %f; %f %f %f; %f %f %f]\n",
                    &cam1[0][0], &cam1[0][1], &cam1[0][2],
                    &cam1[1][0], &cam1[1][1], &cam1[1][2],
                    &cam1[2][0], &cam1[2][1], &cam1[2][2]) != 9)
            {
                ShowErrorMsg("cam1"); return false;
            }

            if (fgets(buffer, sizeof(buffer), FileHandler.get()) == nullptr ||
                sscanf(buffer, "doffs = %f\n", &doffs) != 1)
            {
                ShowErrorMsg("doffs"); return false;
            }

            if (fgets(buffer, sizeof(buffer), FileHandler.get()) == nullptr ||
                sscanf(buffer, "baseline = %f\n", &baseline) != 1)
            {
                ShowErrorMsg("baseline"); return false;
            }

            if (fgets(buffer, sizeof(buffer), FileHandler.get()) == nullptr ||
                sscanf(buffer, "width = %d\n", &imWidth) != 1)
            {
                ShowErrorMsg("width"); return false;
            }

            if (fgets(buffer, sizeof(buffer), FileHandler.get()) == nullptr ||
                sscanf(buffer, "height = %d\n", &imHeight) != 1)
            {
                ShowErrorMsg("height"); return false;
            }

            if (fgets(buffer, sizeof(buffer), FileHandler.get()) == nullptr ||
                sscanf(buffer, "ndisp = %d\n", &ndisp) != 1)
            {
                ShowErrorMsg("ndisp"); return false;
            }

            // following parameters are less important

            if (fgets(buffer, sizeof(buffer), FileHandler.get()) != nullptr)
                sscanf(buffer, "isint = %d\n", &isint);

            if (fgets(buffer, sizeof(buffer), FileHandler.get()) != nullptr)
                sscanf(buffer, "vmin = %d\n", &vmin);

            if (fgets(buffer, sizeof(buffer), FileHandler.get()) != nullptr)
                sscanf(buffer, "vmax = %d\n", &vmax);

            if (fgets(buffer, sizeof(buffer), FileHandler.get()) != nullptr)
                sscanf(buffer, "dyavg = %f\n", &dyavg);

            if (fgets(buffer, sizeof(buffer), FileHandler.get()) != nullptr)
                sscanf(buffer, "dymax = %f\n", &dymax);
        }

        return true;
    }

    friend std::ostream & operator<< (std::ostream & out, const CalibrationParams & clb)
    {
        out.precision(3);
        out << std::fixed;

        out << "cam0     : [" << clb.cam0[0][0] << " " << clb.cam0[0][1] << " " << clb.cam0[0][2] << "; "
            << clb.cam0[1][0] << " " << clb.cam0[1][1] << " " << clb.cam0[1][2] << "; "
            << clb.cam0[2][0] << " " << clb.cam0[2][1] << " " << clb.cam0[2][2] << "]\n";

        out << "cam1     : [" << clb.cam1[0][0] << " " << clb.cam1[0][1] << " " << clb.cam1[0][2] << "; "
            << clb.cam1[1][0] << " " << clb.cam1[1][1] << " " << clb.cam1[1][2] << "; "
            << clb.cam1[2][0] << " " << clb.cam1[2][1] << " " << clb.cam1[2][2] << "]\n";

        out << "doffs    : " << clb.doffs << "\n"
            << "baseline : " << clb.baseline << "\n"
            << "width    : " << clb.imWidth << "\n"
            << "height   : " << clb.imHeight << "\n"
            << "ndisp    : " << clb.ndisp << "\n"
            << "isint    : " << clb.isint << "\n"
            << "vmin     : " << clb.vmin << "\n"
            << "vmax     : " << clb.vmax << "\n"
            << "dyavg    : " << clb.dyavg << "\n"
            << "dymax    : " << clb.dymax << "\n";

        return out;
    }
};

struct Properties
{
    bool enforce;
    int  superpixsz;
    int  compactness;

    double sigma;
    double lowBound;
    double highBound;

    std::string inputDir;
    std::string outputDir;
    std::string calibFileName;

    CalibrationParams calib;

    friend std::ostream & operator<< (std::ostream & out, const Properties & props)
    {
        out << "Input directory : " << props.inputDir
            << "\nOutput directory : " << props.outputDir
            << "\nCalibration file name : " << props.calibFileName
            << "\nSuper pixel size : " << props.superpixsz
            << "\nCalibration params =>\n" << props.calib << "\n";

        return out;
    }
};

#endif