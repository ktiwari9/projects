/**
 * @file:   bagOfFeatures.h
 * @author: Jan Hendriks (dahoc3150 [at] yahoo.com)
 *
 * @date: Created on 31. August 2011, 18:33
 *
 * @brief: Bag of Features / Bag of Words / Bag of Visual Words / Bag of Keypoints implementation using openCV library and openCV internal FLANN with usage example of example data
 *
 * This code uses the in openCV already included FLANN library for faster and uniform processing, accessed via cv::flann opencv2 interface
 * @see: http://opencv-users.1802565.n2.nabble.com/FLANN-kdtree-to-find-k-nearest-neighbors-of-a-point-in-a-pointcloud-td6347494.html
 *
 */

#ifndef BAGOFFEATURES_H
#define BAGOFFEATURES_H

#include "stdlib.h"
#include "fstream"

#include "opencv2/opencv.hpp"
#include "opencv2/ml/ml.hpp"

// FLANN specific
#include "opencv-2.3.1/opencv2/flann/config.h" // for FLANN_VERSION_
#include "opencv-2.3.1/opencv2/flann/flann.hpp"

class bagOfFeatures {

private:
    cv::Mat codeWords; // cluster centers / centroids
    unsigned int numberOfClusters;
    cv::flann::KMeansIndexParams kMeansParameters;

public:

    /**
     * Constructor, sets FLANN parameters
     * @WARNING, @NOTE: Currently seed for random cluster centers is unavailable in openCV FLANN
     * @see http://opencv.willowgarage.com/documentation/cpp/clustering_and_search_in_multi-dimensional_spaces.html
     */
    bagOfFeatures(const int _branching = 2, const int _iterations = 200, const cvflann::flann_centers_init_t _ci = cvflann::CENTERS_RANDOM, const float _cb_index = 0.0) : numberOfClusters(0)  {

        this->kMeansParameters = cv::flann::KMeansIndexParams(
        _branching, // (int)(dimension / 2), // int branching
        _iterations, // int iterations
        _ci, // CENTERS_KMEANSPP, // CENTERS_RANDOM, CENTERS_GONZALES flann_centers_init_t centers_init
        _cb_index); // float cb_index
        printf("Using FLANN library version %s\n", FLANN_VERSION_);
    }

    const inline void setBranching(const int& _branching) {
        this->kMeansParameters.branching = _branching;
    }

    const inline int getBranching() {
        return this->kMeansParameters.branching;
    }

    const inline void setIterations(const int& _iterations) {
        this->kMeansParameters.iterations = _iterations;
    }

    const inline int getIterations() {
        return this->kMeansParameters.iterations;
    }

    const inline void setCentersInitType(const cvflann::flann_centers_init_t& _centers_init) {
        this->kMeansParameters.centers_init = _centers_init;
    }

    const inline cvflann::flann_centers_init_t getCentersInitType() {
        return this->kMeansParameters.centers_init;
    }

    const inline void setCbIndex(const float& _cb_index) {
        this->kMeansParameters.cb_index = _cb_index;
    }

    const inline float getCbIndex() {
        return this->kMeansParameters.cb_index;
    }

    /**
     * Returns the number of clusters
     * @return number of clusters = codebook size
     */
    const inline unsigned int getNumberOfClusters() {
        return this->numberOfClusters;
    }

    /**
     * Generates the FLANN (search) index with the currently set parameters
     * @WARNING When changes to the parameters (e.g. via setter methods) were done, you need to call this in order to acutally *use* the newly set parameters
     * @param _allAvailablePosAndNegSamples input data to generate the search index on
     * @return a flann index data structure
     */
    cv::flann::Index generateIndex(const cv::Mat& _allAvailablePosAndNegSamples) {
        cv::flann::Index_<float> flannIndex(_allAvailablePosAndNegSamples, this->kMeansParameters);
//        cv::flann::Index_< cv::flann::L2<float> > flannIndex(_allAvailablePosAndNegSamples, this->kMeansParameters);
//        cv::flann::Index flannIndex(_allAvailablePosAndNegSamples, this->kMeansParameters);
        return flannIndex;
    }

    /**
     * Loads a mat containing the extracted code words = cluster centers
     * @param _fileName to load mat from
     * @return loaded mat
     */
    cv::Mat loadCodeWords(const std::string& _fileName) {
        cv::Mat cw;
        cv::FileStorage fs(_fileName.c_str(), cv::FileStorage::READ);
        fs["codeWords"] >> cw;
        printf("Loaded Mat of size %d x %d from file '%s'\n", cw.rows, cw.cols, _fileName.c_str());
        return cw;
    }

    /**
     * Saves the mat containing the extracted code words = cluster centers to a file
     * @param _fileName to save to
     * @param _mat containing the code words
     */
    inline void saveCodeWords(const std::string& _fileName, const cv::Mat& _mat) {
        cv::FileStorage fs(_fileName.c_str(), cv::FileStorage::WRITE);
        fs << "codeWords" << _mat;
        printf("Saved %d code words to file '%s'\n", _mat.rows, _fileName.c_str());
    }

    /**
     * k-NN via FLANN to get cluster correspondences for a feature vectors (one feature vector per Mat row)
     * @param flannIndex flann index
     * @param _featureVectors one feature vector per row
     * @param k the k in knn, only the single nearest neighbour is needed for bof
     * @return indices of nearest clusters in the form row=associated samples, col=k nearest neighbors
     */
    cv::Mat getClusterMembershipForFeatures(cv::flann::Index& _flannIndex, const cv::Mat& _featureVectors, const unsigned int k = 1) {
        /// I chose FLANN over opencv KNN or pcl FLANN because the clustering is also done in it, avoid senseless mixing & conversion of different libraries because one will introduce the drawbacks of both libraries
//        printf("Cluster membership association: %d queries for %d knn of dim %d\n", _featureVectors.rows, k, _featureVectors.cols);

        cv::Mat indices = cv::Mat::zeros(_featureVectors.rows, k, CV_32S); // this->codeWords->cols = dimensionality of points
        cv::Mat distances = cv::Mat::zeros(_featureVectors.rows, k, CV_32F); // this->codeWords->cols = dimensionality of points

        // # of checks, tradeoff: lower = faster & less precise, higher = slower & more accurate
        int numberOfChecks = 100; /// @WARNING magic number

        /// do a knn search, @see http://opencv.willowgarage.com/documentation/cpp/fast_approximate_nearest_neighbor_search.html
        _flannIndex.knnSearch(_featureVectors, indices, distances, k, cv::flann::SearchParams(numberOfChecks));
        return indices; // Return the associated nearest cluster indices for each sample, rows refer to samples, cols refer to the knn, #rows = #samples = _featureVectors.rows, #cols = k
    }

    /**
     * Generate the clustering from specified input data (points/point histograms)
     * @param _allAvailablePosAndNegSamples the features (=rows) and their components (=columns)
     * @param numberOfDesiredClusters the desired number of clusters (will be equal or less depending on FLANN), values 200-500 recommended for my real-world high-dimensional data
     */
    cv::Mat generateClusters(const cv::Mat& _allAvailablePosAndNegSamples, const unsigned int _numberOfDesiredClusters = 500) {
        /// The fast way using integrated FLANN library:
//        int dimension = _allAvailablePosAndNegSamples->cols;
        printf("Generating clusters (%d, %d)\n", _numberOfDesiredClusters, _allAvailablePosAndNegSamples.cols);
        cv::Mat cWords = cv::Mat::zeros(_numberOfDesiredClusters, _allAvailablePosAndNegSamples.cols, CV_32F);

        // The function returns the resulting number of actual clusters calculated for a low cluster variety
        this->numberOfClusters = cv::flann::hierarchicalClustering<float, float>(_allAvailablePosAndNegSamples, cWords, this->kMeansParameters);

        printf("Actual number of clusters obtained: %d\n", this->numberOfClusters);
        return cWords;
    }

    /**
     * Generates a single normed histogram over given cluster memberships
     * @param _clustersOfFeaturesFromSingleROI a matrix in the form row=sample, col=associated cluster
     * @return mat, normed histogram
     */
    cv::Mat generateHistogramOfCodeWords(const cv::Mat& _clustersOfFeaturesFromSingleROI) {
        cv::Mat histogram = cv::Mat::zeros(this->numberOfClusters, 1, CV_32F);
        unsigned int accumulator = 0;
        for (; accumulator < _clustersOfFeaturesFromSingleROI.rows; ++accumulator) {
//            printf("Acc: %d, val %d\n", accumulator, _clustersOfFeaturesFromSingleROI.at<int>(accumulator, 0));
            ++histogram.at<float>(_clustersOfFeaturesFromSingleROI.at<int>(accumulator, 0), 0);
//            printf("Cluster association at %d (index %d): %+3.2f\n", accumulator, _clustersOfFeaturesFromSingleROI.at<int>(accumulator, 0), histogram.at<float>(_clustersOfFeaturesFromSingleROI.at<int>(accumulator, 0), 0));
        }
        cv::Mat normedHistogram = cv::Mat(this->numberOfClusters, 1, CV_32F);
        for (unsigned int i = 0; i < histogram.rows; ++i) {
            normedHistogram.at<float>(i,0) = histogram.at<float>(i,0) / (float)accumulator; // L1 norm is slightly faster than the common L2 one
//            printf("Normed histogram at %d: %+3.2f\n", i, normedHistogram.at<float>(i,0));
        }
        return normedHistogram; // normedHistogram.t();
    }

    /**
     * Default destructor
     */
    virtual ~bagOfFeatures() {}

};

#endif  /* BAGOFFEATURES_H */
