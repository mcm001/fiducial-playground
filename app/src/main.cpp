#include "aruco_nano.h"
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <opencv2/aruco/dictionary.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgcodecs.hpp>
#include <fmt/core.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/aruco.hpp>

using namespace aruconano;
using namespace std::chrono;
using namespace std::chrono_literals;

using double_ms_dur = std::chrono::duration<double, std::milli>;

static cv::Mat image_1829px, image_1829px_gray;
static cv::Mat image_3658px, image_3658px_gray;
static constexpr int kWarmupLoops = 10;

std::pair<double_ms_dur, std::vector<Marker>> aruco_v5_detect(const cv::Mat &inputMat) {
	std::vector<Marker> detectedMarkers;

	steady_clock::time_point begin = steady_clock::now();
	auto markers = MarkerDetector::detect(inputMat, 10, TagDicts::APRILTAG_36h11);
	double_ms_dur detectElapsed = steady_clock::now() - begin;

	return {detectElapsed, detectedMarkers};
}


void runTestSuite(bool parallel, bool hires, 
	std::function<double(const cv::Mat&)> detectFunc, std::string detectName, const int iters = 100) {
	cv::Mat colorMat = hires ? image_3658px : image_1829px;
	cv::Mat grayMat = hires ? image_3658px_gray : image_1829px_gray;

	fmt::println("[{}] Loaded image: {} x {}. Paralellization: {}", 
		detectName, colorMat.cols, colorMat.rows, (parallel ? "ON" : "OFF"));
	fmt::println("[{}] Running {} iterations for color and grayscale", detectName, iters);

	cv::setNumThreads(parallel ? -1 : 0);

	double colorMeans = 0.0;
	double grayMeans = 0.0;

	for (int i = 0; i < iters; i++) {
		double time = detectFunc(colorMat);
		colorMeans += time;
	}
	double colorMean = colorMeans / (double)iters;
	fmt::println("[{}] Mean (Color): {:.4f}ms", detectName, colorMean);

	for (int i = 0; i < iters; i++) {
		double time = detectFunc(grayMat);
		grayMeans += time;
	}

	double grayMean = grayMeans / (double)iters;
	double grayDiff = colorMean - grayMean;
	double grayDiffPct = (grayDiff / colorMean) * 100.0;

	fmt::println("[{}] Mean (Gray): {:.4f}ms", detectName, grayMean);
	fmt::println("[{}] Perf diff: {:.4f}ms ({:.2f}%)", detectName, grayDiff, grayDiffPct);
}

struct TestEntry {
	std::string name;
	bool hires;
	bool opencv_parallel;
	std::function<double(const cv::Mat&)> func;
};

struct TestResult {

};

static auto aruco_v5_detectFunc = [](const cv::Mat &input) {
	return aruco_v5_detect(input).first.count();
};

static const cv::Ptr<cv::aruco::Dictionary> dict =
		cv::aruco::getPredefinedDictionary(cv::aruco::DICT_APRILTAG_36h11);

static auto opencv_aruco_detectFunc = [](const cv::Mat &input) {
	std::vector<std::vector<cv::Point2f>> markers;
	std::vector<int> ids;

	// don't measure this - we only care about detector performance
	cv::Mat tmp = input;
	if (input.channels() == 3) { cv::cvtColor(tmp, tmp, cv::COLOR_BGR2GRAY); }

	steady_clock::time_point begin = steady_clock::now();
	cv::aruco::detectMarkers(tmp, dict, markers, ids);
	double_ms_dur detectElapsed = steady_clock::now() - begin;
	return detectElapsed.count();
};

static const std::vector<TestEntry> testEntries = {
	{"aruco_v5_1", false, false, aruco_v5_detectFunc},
	{"aruco_v5_2", true,  false, aruco_v5_detectFunc},
	{"aruco_v5_3", false, true,  aruco_v5_detectFunc},
	{"aruco_v5_4", true,  true,  aruco_v5_detectFunc},
	{"opencv_aruco_1", false, false, opencv_aruco_detectFunc},
	{"opencv_aruco_2", true,  false, opencv_aruco_detectFunc},
	{"opencv_aruco_3", false, true,  opencv_aruco_detectFunc},
	{"opencv_aruco_4", true,  true,  opencv_aruco_detectFunc}
};

int main() {
	// std::cout << cv::getBuildInformation() << std::endl << std::endl;
	image_1829px = cv::imread("image1.jpg");
	image_3658px = cv::imread("image1_2x_res.jpg");
	cv::cvtColor(image_1829px, image_1829px_gray, cv::COLOR_BGR2GRAY);
	cv::cvtColor(image_3658px, image_3658px_gray, cv::COLOR_BGR2GRAY);
	
	for(auto &test : testEntries) {
		runTestSuite(test.opencv_parallel, test.hires, test.func, test.name);
		fmt::println("Completed test {}\n", test.name);
	}
}