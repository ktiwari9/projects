/**
 * @file:   main.cpp
 * @author: Jan Hendriks (dahoc3150 [at] yahoo.com)
 *
 * @date: Created on 12. August 2011, 14:32
 *
 * @brief: Bag of Features / Bag of Words / Bag of Visual Words / Bag of Keypoints implementation using openCV library and openCV internal FLANN with usage example of example data
 *
 * Compile by issuing:
 * g++ `pkg-config --cflags opencv` `pkg-config --libs opencv` -o bof main.cpp bagOfFeatures.h
 *
 * or, for opencv 2.3.1:
 * g++ `pkg-config --cflags opencv-2.3.1` `pkg-config --libs opencv-2.3.1` -o bof main.cpp bagOfFeatures.h
 *
 */

#include <stdlib.h>

#include "bow.h"

using namespace std;
using namespace cv;

/*
gnuplot>
plot './dataPointsWithClusterIds.dat' using 1:2:3 with points lc variable title "Data points", './codeWordsWithId.dat' using 1:2:3 with points pt 5 lc variable title "Cluster Centers"
 */
static void saveForGnuPlot(const cv::Mat& _dataToSave, const std::string _fileName) {
    setlocale(LC_ALL, "C"); // Do not use the system locale
    std::fstream File;
    File.open(_fileName.c_str(), ios::out);
    if (File.good() && File.is_open()) {
        for (unsigned int row = 0; row < _dataToSave.rows; ++row) {
            for (unsigned int col = 0; col < _dataToSave.cols; ++col) {
                File << _dataToSave.at<float>(row,col) << " ";
            }
            File << std::endl;
        }
        File.flush();
        File.close();
    }
}

/**
 * bof simple minimalistic "unit" test
 * @param argc
 * @param argv
 * @return EXIT_SUCCESS | EXIT_FAILURE
 */
int main(int argc, char** argv) {

    bagOfFeatures bof;
    srand ( 3015 );
//    srand ( time(NULL) );

    /// Training part

    unsigned int dimensionality = 2;
    // Synthetically generate data that we use for building up clusters, usually they in some way simulate/inherit a characteristic of the underlying dataset
    cv::Mat features = cv::Mat::zeros(800, dimensionality, CV_32F); // x entries of y-dim data
    int clusterCount = 10; // Generate 10 clusters
    int rowsThatContainOneCluster = (int)((features.rows) / clusterCount);

    cv::Mat tmp = cv::Mat::zeros(1,1,CV_32F); randn(tmp, cv::Scalar::all(0), cv::Scalar::all(1)); /// @NOTE: Seems like a bug, randn() needs one extra run to generate decent data, otherwise there is a fixed relation x-y
    for (int i = 0; i < clusterCount; ++i) {
        cv::Mat currentRowRange = cv::Mat::zeros(rowsThatContainOneCluster, features.cols, features.type());
        cv::Mat xValues = cv::Mat::zeros(rowsThatContainOneCluster, 1, features.type());
        cv::Mat yValues = cv::Mat::zeros(rowsThatContainOneCluster, 1, features.type());
        int currentClusterCenterX = (rand() % 90)+10; // 10..99
        int currentClusterCenterY = (rand() % 90)+10;
        printf("New cluster center: %d,%d\n", currentClusterCenterX, currentClusterCenterY);
        randn(xValues, cv::Scalar::all(currentClusterCenterX), cv::Scalar::all(3)); // mat, mean, stddev
        randn(yValues, cv::Scalar::all(currentClusterCenterY), cv::Scalar::all(3)); // mat, mean, stddev
        for (int x = 0; x < currentRowRange.rows; ++x) {
            features.at<float>(i*rowsThatContainOneCluster+x, 0) = xValues.at<float>(x,0);
            features.at<float>(i*rowsThatContainOneCluster+x, 1) = yValues.at<float>(x,0);
        }

    }

    // The desired number of clusters / code words = codebook size, the actual number will be less than this value depending on the input data and FLANN settings like branching factor, initial cluster center selection & method
    int desiredNumberOfClusterCenters = 10; // 200 - 500 is reasonable for my actual data afterwards

    /// @TODO: Do I need this? Additional index generation influences the cluster calculation?! Why does FLANN do that?
//    cv::flann::Index dataIndex = bof.generateIndex(samples);
//    cv::flann::Index dataIndex2 = bof.generateIndex(samples);

    // Generate cluster centers / centroids from ALL training samples (pos&neg) disregarding the training sample labels
    cv::Mat codeWords = bof.generateClusters(features, desiredNumberOfClusterCenters);

    // Add associated index as last column for plotting
    cv::Mat codeWordsWithId = cv::Mat::zeros(codeWords.rows, codeWords.cols+1, CV_32F);
    for (unsigned int clusterIdx = 0; clusterIdx < codeWords.rows; ++clusterIdx) {
        for (unsigned int cols = 0; cols < codeWords.cols; ++cols) {
            codeWordsWithId.at<float>(clusterIdx, cols) = codeWords.at<float>(clusterIdx, cols);
        }
        codeWordsWithId.at<float>(clusterIdx, codeWords.cols) = clusterIdx+1;
    }
    saveForGnuPlot(codeWordsWithId, string("codeWordsWithId.dat"));

    // Init cluster centers knn search
    cv::flann::Index clusterIndex(bof.generateIndex(codeWords));

    if (codeWords.rows==0) { // Make sure there are cluster centers
        printf("No cluster centers could be generated!\n");
        exit(EXIT_FAILURE);
    }

    // Output the generated cluster centers
    for (unsigned int i = 0; i < bof.getNumberOfClusters(); ++i) {
        printf("Generated cluster with id %d and coordinates (", i);
        for (unsigned int j = 0; j < codeWords.cols-1; ++j) {
            printf("%3.2f, ", codeWords.at<float>(i,j));
        }
        printf("%3.2f)\n", codeWords.at<float>(i,codeWords.cols-1));
    }

    // Append the cluster correspondence for each sample=line as last column
    cv::Mat featuresAndClusterIndex = cv::Mat::zeros(features.rows, features.cols + 1, CV_32F);
    for (unsigned int row = 0; row < features.rows; ++row) {
        cv::Mat clusterMemberships = bof.getClusterMembershipForFeatures(clusterIndex, features.row(row), 1);
        for (unsigned int cols = 0; cols < features.cols; ++cols)
            featuresAndClusterIndex.at<float>(row, cols) = features.at<float>(row, cols);
        featuresAndClusterIndex.at<float>(row, features.cols) = (float)(clusterMemberships.at<int>(0, 0)+1);
    }
    saveForGnuPlot(featuresAndClusterIndex, string("dataPointsWithClusterIds.dat"));

    // Now that the code words have been generated from the input training set, calculate histograms of the code word associations of the several features in each ROI sample containing relevant information about the code word dependencies for corresponding features of each single ROI
    // Example if each 2 features belong to a single sample from a single ROI:
    cv::Mat bags = cv::Mat(50, features.cols, CV_32F);
    for (unsigned int i = 0; i < bags.rows-1; i+=2) {
//        printf("Generating a single bag out of 2 samples, step %d\n", i);
        cv::Mat samplesOfSingleROI = cv::Mat::zeros(2, features.cols, CV_32F);
        samplesOfSingleROI.at<float>(0,0) = features.at<float>(i,0);
        samplesOfSingleROI.at<float>(0,1) = features.at<float>(i,1);
        samplesOfSingleROI.at<float>(1,0) = features.at<float>(i+1,0);
        samplesOfSingleROI.at<float>(1,1) = features.at<float>(i+1,1);
        cv::Mat clusterMemberships = bof.getClusterMembershipForFeatures(clusterIndex, samplesOfSingleROI, 1);
        printf("Approximate nearest cluster membership for sample #%d (value %+3.2f, +%3.2f): #%d (value %+3.2f, %+3.2f) | (value %+3.2f, +%3.2f): #%d (value %+3.2f, %+3.2f)\n", i,
                features.at<float>(i,0), features.at<float>(i,1),
                clusterMemberships.at<int>(0,0),
                codeWords.at<float>(clusterMemberships.at<int>(0,0), 0),
                codeWords.at<float>(clusterMemberships.at<int>(0,0), 1),
                features.at<float>(i+1,0), features.at<float>(i+1,1),
                clusterMemberships.at<int>(1,0),
                codeWords.at<float>(clusterMemberships.at<int>(1,0), 0),
                codeWords.at<float>(clusterMemberships.at<int>(1,0), 1));
//        printf("Unify the 2 samples\n");
        bags.row(i) = bof.generateHistogramOfCodeWords(clusterMemberships); // Now we generate a histogram (or bag if you want) over cluster associations for each ROI containing 2 features
    }
    // For each sample containing several (here: 2) features we now have a sort of "statistic" over code word associations ideally representing the character of the specific sample

    // Corresponding sample labels, one for each sample generated from a single ROI
    cv::Mat sampleLabels = cv::Mat(bags.rows, 1, CV_32S);
    for (unsigned int i = 0; i < sampleLabels.rows; ++i) {
        sampleLabels.at<int>(i,0) = (i < 10 ? 1 : -1);
    }

    /// @TODO: Now we are able to pass the histograms / bofs / bags of each sample for a ROI to a training routine along with the label for the single sample combining several features of a single ROI

    // Simply only save and load the required codeWords Mat (and not the unsupported flann index structure which can be re-built using the parameters and the loaded code words Mat)
    std::string fileName = "bofCodeWords.xml";
    bof.saveCodeWords(fileName, codeWords);

    cv::Mat retrievedCodeWords = bof.loadCodeWords(fileName);
    /// @REMARK: Uncomment below line when used in code later
//    cv::flann::Index retrievedClusterIndex(bof.generateIndex(retrievedCodeWords)); // Init cluster centers knn search

    /// Detection part
    printf("Detection part\n");
    cv::Mat singleSample = cv::Mat::zeros(20, features.cols, CV_32F); // The test query, ROI with 2 key points
    randu(singleSample, cv::Scalar::all(0.0), cv::Scalar(100.0));

    cv::Mat clusterMembership = bof.getClusterMembershipForFeatures(clusterIndex, singleSample, 1);

    for (unsigned int s = 0; s < clusterMembership.rows; ++s) {
        printf("Approximate nearest cluster membership for sample #%d (value %+3.2f, %+3.2f): #%d (value: %+3.2f, %3.2f)\n", s,
                singleSample.at<float>(s,0), singleSample.at<float>(s,1),
                clusterMembership.at<int>(s,0),
                retrievedCodeWords.at<float>(clusterMembership.at<int>(s,0), 0),
                retrievedCodeWords.at<float>(clusterMembership.at<int>(s,0), 1));
    }
    cout << endl; // flush

    cv::Mat histogram = bof.generateHistogramOfCodeWords(clusterMembership); // for the single query with 2 interest points

    // Add the cluster index in front of the normalized bin count
    cv::Mat histogramWithBinId = cv::Mat::zeros(histogram.rows, histogram.cols + 1, CV_32F);
    for (unsigned int bin = 0; bin < histogram.rows; ++bin) {
        histogramWithBinId.at<float>(bin, 0) = bin+1;
        histogramWithBinId.at<float>(bin, 1) = histogram.at<float>(bin, 0);
    }

    saveForGnuPlot(histogramWithBinId, string("histogram.dat"));

    printf("Terminated ok!\n");
    fflush(stdout);

    return(EXIT_SUCCESS);
}
