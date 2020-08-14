#ifndef ARGS_PARSER_H
#define ARGS_PARSER_H

#include <unordered_map>
#include <iostream>
#include "Properties.h"
#include "Utilities.h"

class ArgumentsParser final
{
public:
    ArgumentsParser(int argc, const char** argv)
    {
        for (int i = 0; i < argc - 1; i++)
            if (argv[i][0] == '-')
                argsDict.emplace(&argv[i][1], argv[i + 1]);
    }

    template <typename T>
    bool tryGetValue(std::string settName, T & value) const
    {
        auto foundIter = argsDict.find(settName);
        if (foundIter == argsDict.end())
            return false;

        value = utils::ConvertToValue<T>(foundIter->second);
        return true;
    }

    bool tryGetProperties(Properties & props) const
    {
        if (!tryGetValue("out_dir", props.outputDir))
            return false;

        if (props.outputDir.back() != '\\') {
            props.outputDir += '\\';
        }

        if (!tryGetValue("in_dir", props.inputDir))
            return false;

        if (props.inputDir.back() != '\\') {
            props.inputDir += '\\';
        }

        if (!tryGetValue("spix_sz", props.superpixsz)) {
            props.superpixsz = 2500;
        }

        if (!tryGetValue("compact", props.compactness)) {
            props.compactness = 20;
        }

        if (!tryGetValue("enforce", props.enforce)) {
            props.enforce = true;
        }

        if (!tryGetValue("sigma", props.sigma)) {
            props.sigma = 0.16;
        }

        if (!tryGetValue("low_bound", props.lowBound)) {
            props.lowBound = 0.07;
        }

        if (!tryGetValue("high_bound", props.highBound)) {
            props.highBound = 0.11;
        }

        props.calibFileName = "calib.txt";

        if (!props.calib.read(props.inputDir + props.calibFileName)) {
            return false;
        }

        return true;
    }

    void showProperties() const
    {
        std::cout << "Settings <name> <value> :\n";
        for (const auto & sett : argsDict)
            std::cout << sett.first << " " << sett.second << "\n";
    }

    inline std::string errorMessage() const {
        return "Usage : <exe> "
               "-in_dir <input_dir> "
               "-out_dir <output_dir> "
               "[-spix_sz <super_pix_size>] "
               "[-compact <compactness>] "
               "[-enforce <bool>] "
               "[-sigma <double>] "
               "[-low_bound <double>] "
               "[-high_bound <double>]";
    }

private:
    std::unordered_map<std::string, std::string> argsDict;
};

#endif